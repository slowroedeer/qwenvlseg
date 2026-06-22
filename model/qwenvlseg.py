"""QwenVLSeg: VL + Mask Decoder for referring segmentation.

LLM is completely frozen — relies on Qwen3-VL-Instruct's native 2D grounding
for bbox output. Only mask decoder is trained (~5-8M params).
"""

import json
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .mask_decoder import MaskDecoder


class QwenVLSeg(nn.Module):
    """QwenVLSeg with frozen LLM + trainable mask decoder."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        mask_decoder_cfg: dict | None = None,
    ):
        super().__init__()
        if mask_decoder_cfg is None:
            mask_decoder_cfg = {}

        # ── Load VL model ──
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # Internal components (LLM is frozen — no LoRA)
        inner = self.base_model.model  # Qwen3VLModel inside ConditionalGeneration
        self.vision_encoder = inner.visual
        self.llm_decoder = inner.language_model
        self.lm_head = self.base_model.lm_head

        vc = self.base_model.config.vision_config
        lc = self.llm_decoder.config
        self.llm_dim = lc.hidden_size
        self.vision_dim = vc.hidden_size          # pre-merger
        self.vision_out_dim = vc.out_hidden_size  # post-merger (what LLM receives)
        self.patch_size = vc.patch_size
        self.merge_size = vc.spatial_merge_size

        # ViT hook layer indices (at 1/3 and 2/3 of total layers)
        vit_layers = len(self.vision_encoder.blocks)
        self.vit_hook_layers = [
            vit_layers // 3,
            vit_layers * 2 // 3,
        ]

        # ── Special tokens (for h_mask extraction during training/inference) ──
        special_tokens = {
            "additional_special_tokens": ["<mask_start>", "<mask_token>", "<mask_end>"]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            self.lm_head = self.base_model.lm_head  # re-fetch after resize
        self.mask_start_id = self.tokenizer.convert_tokens_to_ids("<mask_start>")
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<mask_token>")
        self.mask_end_id = self.tokenizer.convert_tokens_to_ids("<mask_end>")

        # ── ViT hooks ──
        self._vit_features = {}
        for idx in self.vit_hook_layers:
            block = self.vision_encoder.blocks[idx]
            block.register_forward_hook(self._make_hook(idx))

        # ── Mask Decoder (only trainable component) ──
        self.mask_decoder = MaskDecoder(
            hidden_dim=mask_decoder_cfg.get("hidden_dim", 256),
            vit_channels=[self.vision_dim, self.vision_dim, self.vision_out_dim],
            llm_dim=self.llm_dim,
            num_transformer_layers=mask_decoder_cfg.get("num_transformer_layers", 2),
            num_heads=mask_decoder_cfg.get("num_heads", 8),
            mask_stride=mask_decoder_cfg.get("mask_stride", 4),
        )

        # Freeze ViT and LLM
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_decoder.parameters():
            param.requires_grad = False

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self._vit_features[layer_idx] = output
        return hook

    def _run_vit(self, pixel_values, image_grid_thw):
        """Run ViT, return vit features + LLM image embeddings + per-image counts."""
        B = image_grid_thw.shape[0]

        merged_out = self.vision_encoder(pixel_values, grid_thw=image_grid_thw)
        pre_seq = merged_out.last_hidden_state  # (sum N_pre, vision_dim)
        post_seq = merged_out.pooler_output     # (sum N_post, vision_out_dim)

        if post_seq.shape[-1] > self.vision_out_dim:
            post_seq = post_seq[..., :self.vision_out_dim]

        hw_list = [(int(thw[1].item()), int(thw[2].item())) for thw in image_grid_thw]
        pre_counts = [h * w for h, w in hw_list]
        post_counts = [(h // self.merge_size) * (w // self.merge_size) for h, w in hw_list]

        all_features = [[], [], []]

        for b in range(B):
            h, w = hw_list[b]
            n_pre = pre_counts[b]
            n_post = post_counts[b]
            pre_start = sum(pre_counts[:b])
            post_start = sum(post_counts[:b])

            for i, layer_idx in enumerate(self.vit_hook_layers):
                feat_seq = self._vit_features[layer_idx]
                feat_b = feat_seq[pre_start:pre_start + n_pre]
                feat_2d = feat_b.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
                all_features[i].append(feat_2d)

            post_b = post_seq[post_start:post_start + n_post]
            h_m, w_m = h // self.merge_size, w // self.merge_size
            post_2d = post_b.reshape(h_m, w_m, -1).permute(2, 0, 1).unsqueeze(0)
            all_features[2].append(post_2d)

        vit_features = [torch.cat(f_list, dim=0) for f_list in all_features]
        return vit_features, post_seq, post_counts

    def _build_inputs_embeds(self, input_ids, visual_embeds_flat, image_grid_thw):
        """Replace image token positions in embeddings with visual features."""
        B = input_ids.shape[0]
        device = input_ids.device
        embed_layer = self.llm_decoder.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        img_token_id = self.base_model.config.image_token_id
        hw_list = [(int(thw[1].item()), int(thw[2].item())) for thw in image_grid_thw]
        post_counts = [(h // self.merge_size) * (w // self.merge_size) for h, w in hw_list]

        offset = 0
        for b in range(B):
            n_img_tokens = post_counts[b]
            img_positions = (input_ids[b] == img_token_id).nonzero(as_tuple=True)[0]
            if img_positions.shape[0] == n_img_tokens:
                inputs_embeds[b, img_positions] = visual_embeds_flat[
                    offset:offset + n_img_tokens
                ]
            else:
                import os as _os
                if _os.environ.get("QWENVLSEG_DEBUG"):
                    print(f"[DEBUG] BuildEmbeds: found {img_positions.shape[0]} img tokens, expected {n_img_tokens}. img_token_id={img_token_id}")
                    print(f"[DEBUG] image_grid_thw={image_grid_thw}")
            offset += n_img_tokens

        return inputs_embeds

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        self.llm_decoder.eval()
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        orig_images: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        gt_bbox: torch.Tensor | None = None,
    ) -> dict:
        """Training forward pass. LLM frozen, only mask decoder trained."""
        device = pixel_values.device
        B = image_grid_thw.shape[0]

        # 1. ViT: multi-scale features + merged visual embeddings
        vit_features, visual_embeds_flat, post_counts = self._run_vit(pixel_values, image_grid_thw)

        # 2. Build input embeddings with visual features
        inputs_embeds = self._build_inputs_embeds(
            input_ids, visual_embeds_flat, image_grid_thw
        )

        # 3. LLM forward (frozen, no gradient)
        llm_out = self.llm_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = llm_out.hidden_states[-1]  # (B, seq_len, llm_dim)

        # 4. Extract h_mask from <mask_token> position
        mask_token_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=False)
        h_mask_list = []
        for b in range(B):
            m_pos = mask_token_positions[mask_token_positions[:, 0] == b]
            if m_pos.shape[0] > 0:
                h_mask_list.append(hidden_states[b, m_pos[0, 1]])
            else:
                h_mask_list.append(hidden_states[b, -1])
        h_mask = torch.stack(h_mask_list)  # (B, llm_dim)

        # 5. Extract T_mm: LLM-processed image embeddings (Eq.5)
        img_token_id = self.base_model.config.image_token_id
        T_mm_list = []
        for b in range(B):
            n_post = post_counts[b]
            img_pos = (input_ids[b] == img_token_id).nonzero(as_tuple=True)[0]
            if img_pos.shape[0] == n_post:
                T_mm_list.append(hidden_states[b, img_pos])
            else:
                T_mm_list.append(hidden_states[b, -n_post:])
        T_mm_embeds = torch.cat(T_mm_list, dim=0)  # (sum n_post, llm_dim)

        # 6. Mask decoder with noisy GT bbox during training (robustness to LLM bbox errors)
        if self.training and gt_bbox is not None:
            # Per-corner independent noise: ±10% of bbox dimensions
            w = gt_bbox[:, 2] - gt_bbox[:, 0]  # (B,)
            h = gt_bbox[:, 3] - gt_bbox[:, 1]  # (B,)
            noise = (torch.rand(B, 4, device=device) * 2 - 1) * 0.1  # [-0.1, 0.1]
            noise[:, 0] *= w
            noise[:, 2] *= w
            noise[:, 1] *= h
            noise[:, 3] *= h
            bbox = gt_bbox + noise
            bbox = bbox.clamp(0, 2000)  # conservative upper bound
            bbox[:, 2] = torch.maximum(bbox[:, 2], bbox[:, 0] + 10)
            bbox[:, 3] = torch.maximum(bbox[:, 3], bbox[:, 1] + 10)
        else:
            bbox = gt_bbox if gt_bbox is not None else torch.zeros(B, 4, device=device)
        mask_out = self.mask_decoder(
            vit_features=vit_features,
            h_mask=h_mask,
            bbox=bbox,
            image=orig_images,
            visual_embeds=T_mm_embeds,
            post_counts=post_counts,
        )
        mask_logits = mask_out['mask_logits']

        # 7. Segmentation loss only
        loss_dict = {}

        if gt_mask is not None:
            mask_up = F.interpolate(
                mask_logits, size=gt_mask.shape[-2:],
                mode='bilinear', align_corners=False,
            )
            valid = (gt_mask != -100).float()
            gt_bin = gt_mask.clamp(min=0).float()

            logits = mask_up.squeeze(1)
            bce = F.softplus(logits) - logits * gt_bin
            bce = (bce * valid).sum() / (valid.sum() + 1e-6)

            pred = torch.sigmoid(mask_up.squeeze(1)) * valid
            dice = dice_loss(pred, gt_bin * valid)

            seg_loss = bce + dice
            loss_dict['seg_loss'] = seg_loss
            loss_dict['bce_loss'] = bce
            loss_dict['dice_loss'] = dice
        else:
            seg_loss = torch.tensor(0.0, device=device)
            loss_dict['seg_loss'] = seg_loss

        loss_dict['loss'] = seg_loss

        return {
            **loss_dict,
            'mask_logits': mask_logits,
            'mask_logits_1': mask_out.get('mask_logits_1'),
            'iou_scores': mask_out.get('iou_scores'),
        }

    @torch.no_grad()
    def generate_and_segment(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        prompt_text: str,
        max_new_tokens: int = 256,
        image_size: int = 512,
    ) -> dict:
        """Inference: LLM generates bbox JSON → parse bbox → extract h_mask → mask decode.

        Two-pass approach:
          Pass 1: LLM generates native JSON (autoregressive, KV-cached) → parse bbox
          Pass 2: Insert mask tokens into generated JSON → LLM forward → extract h_mask
        """
        device = pixel_values.device

        # 1. Build prompt
        system_msg = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        user_msg = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{prompt_text}<|im_end|>\n"
        assistant_prefix = "<|im_start|>assistant\n"
        full_prompt = system_msg + user_msg + assistant_prefix

        # Convert tensor (1,3,H,W) in [0,1] → PIL for processor (expects PIL)
        img_np = (pixel_values.squeeze(0).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        img_pil = Image.fromarray(img_np)

        inputs = self.processor(
            text=[full_prompt],
            images=[img_pil],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 2. ViT
        vit_features, visual_embeds_flat, post_counts = self._run_vit(
            inputs['pixel_values'], inputs['image_grid_thw']
        )

        # ═══ Pass 1: HF native generate (works correctly after resize_token_embeddings) ═══
        gen_out = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prompt_len = inputs['input_ids'].shape[1]
        generated_ids = gen_out[0, prompt_len:].tolist()

        # 5. Decode and parse ALL bboxes from native JSON
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        bboxes = self._parse_bboxes_from_text(generated_text)

        if not bboxes:
            return {
                'mask': None,
                'mask_logits': None,
                'bboxes': [],
                'text_output': generated_text,
                'iou_score': 0.0,
            }

        # ═══ Pass 2: Insert mask tokens → forward → extract h_masks ═══
        if not generated_text.startswith('\n'):
            generated_text = '\n' + generated_text
        extraction_text = self._insert_mask_tokens(generated_text)
        extraction_prompt = full_prompt + extraction_text + "<|im_end|>"

        ext_inputs = self.processor(
            text=[extraction_prompt],
            images=[img_pil],
            return_tensors="pt",
        )
        ext_inputs = {k: v.to(device) for k, v in ext_inputs.items()}

        ext_embeds = self._build_inputs_embeds(
            ext_inputs['input_ids'], visual_embeds_flat, inputs['image_grid_thw']
        )

        ext_out = self.llm_decoder(
            inputs_embeds=ext_embeds,
            attention_mask=ext_inputs['attention_mask'],
            output_hidden_states=True,
        )
        ext_hidden = ext_out.hidden_states[-1]  # (1, total_len, llm_dim)

        # Extract all <mask_token> positions (one per instance)
        ext_ids = ext_inputs['input_ids']
        mask_positions = (ext_ids[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.shape[0] == 0:
            mask_positions = torch.tensor([ext_ids.shape[1] - 1], device=device)

        # Extract T_mm from prompt image tokens
        img_token_id = self.base_model.config.image_token_id
        img_pos_ext = (ext_ids[0] == img_token_id).nonzero(as_tuple=True)[0]
        T_mm_embeds = ext_hidden[0, img_pos_ext]

        # 8. Mask decoder: run per instance, merge via max
        masks_per_instance = []
        iou_scores = []
        n_instances = min(len(bboxes), mask_positions.shape[0])

        for i in range(n_instances):
            h_mask_i = ext_hidden[0, mask_positions[i]]
            bbox_tensor = torch.tensor([bboxes[i]], device=device, dtype=torch.float32)
            mask_out = self.mask_decoder(
                vit_features=vit_features,
                h_mask=h_mask_i.unsqueeze(0),
                bbox=bbox_tensor,
                image=pixel_values,
                visual_embeds=T_mm_embeds,
                post_counts=post_counts,
            )
            masks_per_instance.append(mask_out['mask_logits'])
            iou_scores.append(mask_out['iou_scores'].item())

        # Merge: per-pixel max over all instance masks
        all_masks = torch.cat(masks_per_instance, dim=0)  # (N, 1, H, W)
        mask_logits = all_masks.max(dim=0, keepdim=True)[0]  # (1, 1, H, W)

        mask_logits = F.interpolate(
            mask_logits, size=(image_size, image_size),
            mode='bilinear', align_corners=False,
        )
        mask_pred = (torch.sigmoid(mask_logits) > 0.5).float()

        return {
            'mask': mask_pred.squeeze().cpu().numpy(),
            'mask_logits': mask_logits.squeeze().cpu().numpy(),
            'bboxes': bboxes,
            'text_output': generated_text,
            'iou_score': np.mean(iou_scores) if iou_scores else 0.0,
        }

    def _insert_mask_tokens(self, generated_text: str) -> str:
        """Insert mask field into ALL objects in generated JSON — preserves exact formatting."""
        mask_field = f'"mask": "<mask_start><mask_token><mask_end>"'
        result = re.sub(
            r'("label":\s*"[^"]+")}',
            rf'\1, {mask_field}}}',
            generated_text,
        )
        return result

    def _parse_bboxes_from_text(self, text: str) -> list:
        """Parse ALL bboxes from generated JSON. Returns list of [x1,y1,x2,y2]."""
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                bboxes = []
                for item in data:
                    bbox = item.get("bbox_2d")
                    if bbox and len(bbox) == 4:
                        bboxes.append([int(x) for x in bbox])
                if bboxes:
                    return bboxes
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        # Fallback: regex match single bbox
        bbox_match = re.search(r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
        if bbox_match:
            return [[int(x) for x in bbox_match.groups()]]
        return []

    def save_checkpoint(self, path: str):
        torch.save({
            'mask_decoder': self.mask_decoder.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.mask_decoder.load_state_dict(ckpt['mask_decoder'], strict=False)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()
