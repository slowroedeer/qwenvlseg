"""Box-Guided Mask Decoder — from Qwen3-VL-Seg paper.

Components (Section 3.2):
  1. Multi-scale Spatial Feature Injection (3.2.1)
  2. Spatial-Semantic Query Construction (3.2.2)
  3. Box-Guided High-Resolution Pixel Fusion (3.2.3)
  4. Iterative Mask-Aware Query Refinement (3.2.4)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────
# 1. Spatial Feature Injector (Eq. 2, 3)
# ────────────────────────────────────────────

class SpatialFeatureInjector(nn.Module):
    """Adapts a ViT feature map: 1×1 Conv → GroupNorm → DWConv → residual."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, groups=out_channels
        )
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.s = nn.Parameter(torch.tensor(1e-3))  # near-zero init for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        x0 = self.proj(x)                           # Eq.2
        out = x0 + self.s * F.gelu(self.dwconv(self.norm(x0)))  # Eq.3
        return out


# ────────────────────────────────────────────
# 2. Box Encoder (Eq. 7, 8)
# ────────────────────────────────────────────

class BoxEncoder(nn.Module):
    """Encodes bbox [x1,y1,x2,y2] with Fourier PE + MLP."""

    def __init__(self, hidden_dim: int = 256, num_freqs: int = 32):
        super().__init__()
        # 4 coords (x1,y1,log_w,log_h), each encoded with num_freqs freqs → 4*(2*num_freqs+1)
        self.num_freqs = num_freqs
        input_dim = 4 * (2 * num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """Encode bounding box.

        Args:
            bbox: (B, 4) — [x1, y1, x2, y2] in 0-1000 normalized coords

        Returns:
            (B, hidden_dim)
        """
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)

        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        w = (x2 - x1).clamp(min=1)
        h = (y2 - y1).clamp(min=1)

        # Fourier PE on x1, y1, log_w, log_h (Eq.8)
        feats = [
            fourier_encode(x1, self.num_freqs, max_val=1000.0),
            fourier_encode(y1, self.num_freqs, max_val=1000.0),
            fourier_encode(0.2 * torch.log(w) + 0.5, self.num_freqs, max_val=1.0),
            fourier_encode(0.2 * torch.log(h) + 0.5, self.num_freqs, max_val=1.0),
        ]
        e_box = torch.cat(feats, dim=-1)  # (B, input_dim)
        return self.mlp(e_box)


def fourier_encode(x: torch.Tensor, num_freqs: int, max_val: float = 1.0) -> torch.Tensor:
    """Fourier positional encoding for scalar values."""
    scales = 2 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    scales = scales * math.pi / max_val
    x = x.reshape(-1, 1, 1) * scales.view(1, 1, -1)  # (B, 1, num_freqs)
    return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)  # (B, 2*num_freqs)


# ────────────────────────────────────────────
# 3. Transformer Decoder Layer
# ────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    """Self-attention + cross-attention to memory + FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, memory: torch.Tensor):
        """Args:
            query: (B, 1, d_model) — object query
            memory: (B, N_mem, d_model) — flattened memory features
        """
        q = self.norm1(query)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm2(q)
        q = q + self.cross_attn(q, memory, memory)[0]
        q = self.norm3(q)
        q = q + self.ffn(q)
        return q


# ────────────────────────────────────────────
# 4. Main Mask Decoder
# ────────────────────────────────────────────

class MaskDecoder(nn.Module):
    """Box-guided mask decoder as described in Qwen3-VL-Seg Section 3.2.

    Args:
        hidden_dim: decoder hidden dimension (default 256)
        vit_channels: list of input channels from each ViT scale
        num_transformer_layers: number of transformer decoder layers
        num_heads: number of attention heads
        mask_stride: output mask is at this stride relative to input, then upsampled
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        vit_channels: list = None,
        llm_dim: int = 2048,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        mask_stride: int = 4,
    ):
        super().__init__()
        if vit_channels is None:
            vit_channels = [1024, 1024, 2048]
        self.hidden_dim = hidden_dim
        self.mask_stride = mask_stride

        # ── 3.2.1 Multi-scale Spatial Feature Injection ──
        self.spatial_injectors = nn.ModuleList([
            SpatialFeatureInjector(in_c, hidden_dim) for in_c in vit_channels
        ])
        # Conv fusion after concat
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * len(vit_channels), hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

        # Project multimodal visual embeddings (T_mm) to 2D
        self.mm_projector = nn.Linear(llm_dim, hidden_dim)

        # ── 3.2.2 Query Construction ──
        self.box_encoder = BoxEncoder(hidden_dim=hidden_dim)
        self.seg_projector = nn.Linear(llm_dim, hidden_dim)

        # ── Transformer Decoder (custom layers, avoids PyTorch version issues) ──
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4)
            for _ in range(num_transformer_layers)
        ])

        # ── 3.2.3 Pixel Fusion ──
        # Lightweight CNN stem for shallow features
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
        )
        # PixelShuffle upsample: 2× two-stage → 4× total
        self.upsample1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # H×2, W×2, channels/4
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # H×2, W×2, channels/4
        )

        # ── 3.2.4 Mask Prediction (dynamic convolution) ──
        # Query → dynamic kernel for mask prediction
        self.dynamic_kernel_1 = nn.Linear(hidden_dim, hidden_dim)
        self.dynamic_kernel_2 = nn.Linear(hidden_dim, hidden_dim)

        # ── Refinement projector ──
        self.ref_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── IoU head ──
        self.iou_head = nn.Linear(hidden_dim, 1)

        # ── Learnable 2D positional encoding for memory ──
        self.pos_embed: nn.Parameter | None = None

        # ── Final conv head (on pixel features → mask logits) ──
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def _get_pos_embed(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Get or create 2D positional encoding for memory."""
        if self.pos_embed is None or self.pos_embed.shape[-2:] != (h, w):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.hidden_dim, h, w, device=device)
            )
            nn.init.normal_(self.pos_embed)
        return self.pos_embed

    def build_soft_gate(self, bbox, h, w, alpha=20.0):
        """Build soft spatial gate from bbox (Eq.12)."""
        B = bbox.shape[0]
        device = bbox.device

        # Enlarge box by 15% (Eq.12: x1', y1', x2', y2')
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        bw, bh = (x2 - x1).clamp(min=1), (y2 - y1).clamp(min=1)
        expand_w, expand_h = bw * 0.15, bh * 0.15
        x1_e = (x1 - expand_w).clamp(min=0)
        y1_e = (y1 - expand_h).clamp(min=0)
        x2_e = (x2 + expand_w).clamp(max=1000)
        y2_e = (y2 + expand_h).clamp(max=1000)

        # Coordinate grids in 0-1000 scale
        y_coords = torch.linspace(0, 1000, h, device=device).view(1, 1, h, 1)
        x_coords = torch.linspace(0, 1000, w, device=device).view(1, 1, 1, w)

        # Four sigmoid terms: σ(α(x - x1'))·σ(α(x2' - x))·σ(α(y - y1'))·σ(α(y2' - y))
        gate = (
            torch.sigmoid(alpha * (x_coords - x1_e.view(B, 1, 1, 1)))
            * torch.sigmoid(alpha * (x2_e.view(B, 1, 1, 1) - x_coords))
            * torch.sigmoid(alpha * (y_coords - y1_e.view(B, 1, 1, 1)))
            * torch.sigmoid(alpha * (y2_e.view(B, 1, 1, 1) - y_coords))
        )

        return gate  # (B, 1, H, W)

    def forward(
        self,
        vit_features: list[torch.Tensor],
        h_mask: torch.Tensor,
        bbox: torch.Tensor,
        image: torch.Tensor | None = None,
        visual_embeds: torch.Tensor | None = None,
        post_counts: list | None = None,
    ) -> dict:
        """Forward pass.

        Args:
            vit_features: list of (B, C_i, H_i, W_i) from ViT layers
            h_mask: (B, llm_dim) — semantic query from LLM (<mask_token> hidden state)
            bbox: (B, 4) — [x1,y1,x2,y2] in 0-1000 normalized coords
            image: (B, 3, H_img, W_img) — original image for CNN stem
            visual_embeds: (sum N_post, llm_dim) — LLM-processed image embeddings (T_mm,
                           Eq.5). None for backward compatibility.
            post_counts: list[int] — number of merged image tokens per sample

        Returns:
            dict with mask_logits, mask_logits_1, iou_scores, q_decoded, q_refined
        """
        B = h_mask.shape[0]
        device = h_mask.device

        # ── 3.2.1 Multi-scale Feature Injection ──
        adapted = []
        for feat, injector in zip(vit_features, self.spatial_injectors):
            adapted.append(injector(feat))  # (B, hidden_dim, H_i, W_i)

        # Resize all to the smallest spatial size and concat
        target_h, target_w = adapted[-1].shape[-2:]
        aligned = []
        for feat in adapted[:-1]:
            aligned.append(F.interpolate(feat, size=(target_h, target_w), mode='bilinear'))
        aligned.append(adapted[-1])
        F_fuse = self.fuse_conv(torch.cat(aligned, dim=1))  # (B, hidden_dim, H_f, W_f)
        H_f, W_f = F_fuse.shape[-2:]

        # ── 3.2.2 Query Construction ──
        e_box = self.box_encoder(bbox)                      # (B, hidden_dim)
        q_seg = self.seg_projector(h_mask)                  # (B, hidden_dim)
        q_init = F.layer_norm(q_seg + e_box, [q_seg.shape[-1]])  # Eq.9
        q_init = q_init.unsqueeze(1)  # (B, 1, hidden_dim)

        # ── Memory Construction (Eq.5, 6) ──
        # Eq.5: project multimodal visual embeddings T_mm → 2D
        if visual_embeds is not None and post_counts is not None:
            T_mm_proj = self.mm_projector(visual_embeds)  # (sum N_post, hidden_dim)
            T_mm_2d_list = []
            offset = 0
            for b in range(B):
                n_post = post_counts[b]
                emb_b = T_mm_proj[offset:offset + n_post]  # (n_post, hidden_dim)
                h_p, w_p = int(math.sqrt(n_post)), int(math.sqrt(n_post))
                T_mm_2d_list.append(emb_b.permute(1, 0).reshape(1, self.hidden_dim, h_p, w_p))
                offset += n_post
            T_mm_2d = torch.cat(T_mm_2d_list, dim=0)  # (B, hidden_dim, H_f, W_f)
        else:
            T_mm_2d = torch.zeros_like(F_fuse)

        # Eq.6: F_mem = T_mm + F_fuse + P_mem
        P_mem = self._get_pos_embed(H_f, W_f, device)
        F_mem = T_mm_2d + F_fuse + P_mem
        mem_flat = F_mem.flatten(2).permute(0, 2, 1)  # (B, H_f*W_f, hidden_dim)

        # ── Transformer Decoder ──
        q = q_init  # (B, 1, hidden_dim)
        for layer in self.transformer_layers:
            q = layer(q, mem_flat)
        q_decoded = q  # (B, 1, hidden_dim)

        # ── 3.2.3 Pixel Fusion ──
        # Upsample fused features 4×
        F_up = self.upsample1(F_fuse)  # 2×
        H_up1, W_up1 = F_up.shape[-2:]
        F_up = self.upsample2(F_up)    # 4× total

        # Soft gate from bbox
        gate = self.build_soft_gate(bbox, F_up.shape[-2], F_up.shape[-1])  # (B, 1, H_up, W_up)

        # Shallow CNN features (from original image, stride=4)
        if image is not None:
            F_cnn = self.cnn_stem(image)  # (B, hidden_dim, H/4, W/4)
            # Pad/crop to match F_up spatial size
            if F_cnn.shape[-2:] != F_up.shape[-2:]:
                F_cnn = F.interpolate(F_cnn, size=F_up.shape[-2:], mode='bilinear')
            F_pixel = F_up + gate * F_cnn  # Eq.14
        else:
            F_pixel = F_up

        # ── First-pass mask prediction (dynamic conv) ──
        # Generate dynamic kernel weights from query (normalized to prevent explosion)
        dyn_k1 = F.layer_norm(
            self.dynamic_kernel_1(q_decoded.squeeze(1)), [self.hidden_dim]
        )
        F_mod1 = F_pixel * dyn_k1.unsqueeze(-1).unsqueeze(-1)
        mask_logits_1 = self.mask_conv(F_mod1)  # (B, 1, H_up, W_up)

        # ── 3.2.4 Iterative Refinement ──
        soft_mask = torch.sigmoid(mask_logits_1)  # (B, 1, H_up, W_up)
        eps = 1e-6
        # Target-aware feature pooling (Eq.16)
        F_tar = (soft_mask * F_pixel).sum(dim=[2, 3]) / (soft_mask.sum(dim=[2, 3]) + eps)
        F_tar = self.ref_projector(F_tar)  # (B, hidden_dim)
        q_refined = F.layer_norm(
            q_decoded.squeeze(1) + F_tar, [q_decoded.shape[-1]]
        ).unsqueeze(1)  # Eq.17

        # Second-pass mask (Eq.18)
        q = q_refined  # (B, 1, hidden_dim)
        for layer in self.transformer_layers:
            q = layer(q, mem_flat)
        q_refined = q
        dyn_k2 = F.layer_norm(
            self.dynamic_kernel_2(q_refined.squeeze(1)), [self.hidden_dim]
        )
        F_mod2 = F_pixel * dyn_k2.unsqueeze(-1).unsqueeze(-1)
        mask_logits_2 = self.mask_conv(F_mod2)  # (B, 1, H_up, W_up)

        # IoU score (Eq.20)
        iou_scores = self.iou_head(q_refined.squeeze(1))  # (B, 1)

        return {
            'mask_logits': mask_logits_2,
            'mask_logits_1': mask_logits_1,
            'iou_scores': iou_scores,
            'q_decoded': q_decoded,
            'q_refined': q_refined,
        }
