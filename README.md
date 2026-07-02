# QwenVLSeg — Lightweight Reproduction of Qwen3-VL-Seg

Lightweight reproduction of **Qwen3-VL-Seg** (May 2026). Extends Qwen3-VL's multimodal understanding to pixel-level segmentation by attaching a box-guided mask decoder to the frozen VLM.

- **LLM**: Qwen3.5-0.8B (completely frozen)
- **Mask Decoder**: ~11.6M params, full training
- **Data**: Pascal VOC 2012, all 20 classes (also supports single-class)
- **GPU**: RTX 4080 SUPER (32GB)
- **Training time**: ~32 min for 20 epochs (2157 samples, batch=8, AMP bf16)

> Also supports Qwen3-VL-2B-Instruct — swap `config.yaml` model name to switch.

## Architecture

```
Image → ViT (frozen) → multi-scale features [layer 1/3, 2/3, pooler_output]
                              ↓
Prompt → LLM (frozen) → native JSON {"bbox_2d": [...], "label": "person"}
                              ↓
                    parse bbox  ←  insert mask tokens → LLM forward → h_mask
                              ↓                          ↓
                         Mask Decoder (box-guided) → binary mask
```

LLM is completely frozen — relies on Qwen3.5's native 2D grounding for bbox output. The mask decoder is the only trainable component.

## Setup

```bash
pip install -r requirements.txt
pip install modelscope  # for faster model download on AutoDL
```

Edit `config.yaml` to set your data path.

## Workflow

### 1. Smoke test

```bash
python test_forward.py
```

Verifies model loading, ViT shapes, and forward pass.

### 2. Train

```bash
python train.py --config config.yaml
```

Trains only the mask decoder (~11.6M params). LLM and ViT are frozen. Training uses noisy GT bbox (±10%) for robustness to LLM bbox errors.

Key config in `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen3.5-0.8B"
mask_decoder:
  model_cache_dir: "/root/autodl-tmp/model"   # ModelScope download cache
  hidden_dim: 256
  num_transformer_layers: 2
data:
  root: "/path/to/VOC2012"
  categories: "all"            # "all" = all 20 VOC classes, or ["person", "car"]
  min_mask_pixels: 100
training:
  max_epochs: 20
  batch_size: 8
  learning_rate: 1.0e-4
```

### 3. Evaluate

```bash
# Full val set — multi-class prompt (all 20 classes, 1449 images)
python eval.py --checkpoint /path/to/best_checkpoint.pt --config config.yaml --max-samples 0

# GT bbox upper bound (per-class GT bbox, no VLM generation)
python eval_gt_bbox.py --checkpoint /path/to/best_checkpoint.pt --config config.yaml

# Single-class eval (backward compat, faster + more accurate)
python eval.py --checkpoint /path/to/best_checkpoint.pt --config config.yaml --category person
```

Metrics: mIoU, cIoU, Mean Dice, P@0.5, P@0.7, per-class breakdown (aligned with Qwen3-VL-Seg paper).

### 4. Single image inference

```bash
python inference.py \
  --image /path/to/image.jpg \
  --category person \
  --checkpoint /path/to/best_checkpoint.pt \
  --output mask.png \
  --vis overlay.png
```

### 5. Visualization (batch)

```bash
python visualize.py \
  --checkpoint /path/to/best_checkpoint.pt \
  --num-samples 12 \
  --output-dir /root/autodl-tmp/report_viz
```

Generates 4-panel comparison images (Input | GT | Prediction | Binary Mask).

## Results

### 20-Class Training (Qwen3.5-0.8B, 20 epochs)

1449 val images, multi-class prompt. Mask decoder trains on all 20 VOC classes simultaneously.

**Overall**: mIoU 0.5719 (LLM bbox), 0.7517 (GT bbox upper bound)

LLM mIoU by class (multi-class prompt):

| | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM | .710 | .383 | .775 | .617 | .380 | .600 | .471 | .799 | .204 | .767 | .236 | .762 | .754 | .672 | .500 | .226 | .771 | .547 | .764 | .499 |
| GT | .816 | .508 | .820 | .781 | .624 | .873 | .752 | .866 | .514 | .863 | .701 | .828 | .803 | .808 | .765 | .487 | .864 | .738 | .867 | .756 |
| Δ | -.11 | -.13 | -.05 | -.16 | -.24 | -.27 | -.28 | -.07 | -.31 | -.10 | -.47 | -.07 | -.05 | -.14 | -.27 | -.26 | -.09 | -.19 | -.10 | -.26 |

**Key finding**: Animals (bird/horse/dog/cat, gap <0.10) are well detected as they typically appear alone. Co-occurring classes (diningtable+chair gap 0.31-0.47, person+car gap 0.27-0.28) show large gaps — the VLM misses instances or mislabels them under the long 20-class prompt. Per-class prompt evaluation (see below) eliminates this gap: person alone scores 0.73 vs GT 0.76 (gap only 0.03).

### Person Only — Single-class Prompt (440 val samples)

Per-class prompt `"segment person"`. VLM grounding is highly precise with single-class prompts.

| Metric | Qwen3.5-0.8B LLM | Qwen3.5-0.8B GT | Qwen3-VL-2B LLM | Qwen3-VL-2B GT |
|--------|-------------------|-------------------|-----------------|-----------------|
| mIoU | **0.7343** | 0.7588 | **0.7539** | 0.7975 |
| cIoU | **0.8141** | 0.8300 | **0.8159** | 0.8592 |
| Mean Dice | **0.8267** | 0.8491 | **0.8412** | 0.8764 |
| P@0.5 | **0.9000** | 0.9409 | **0.8864** | 0.9568 |
| P@0.7 | **0.7091** | 0.7273 | **0.7341** | 0.8455 |
| Bbox parse rate | **100%** | — | **100%** | — |

The 0.8B model achieves 97% of the 2B model's mIoU at 40% of the LLM parameters. Both have very small LLM-GT gaps (0.02-0.04), showing grounding is precise. 2B's higher GT ceiling (0.80 vs 0.76) comes from its larger ViT. The 0.8B's GT ceiling improved from 0.759 to 0.765 with 20-class training (see above) — diverse bbox shapes benefit the mask decoder.

## Key Design Decisions

- **LLM completely frozen** — Qwen3-VL-2B and Qwen3.5 both have native 2D grounding from instruction tuning. LoRA training was found to destroy this capability (see `debugnotes.md`).
- **Two-pass inference** — Pass 1: LLM generates bbox JSON. Pass 2: insert mask tokens, forward LLM to extract h_mask.
- **Bbox noise training** — ±10% perturbation on GT bbox bridges the gap between precise GT and approximate LLM bboxes.
- **Multi-instance max merge** — Per-instance mask decode + pixel-wise max, improving coverage from 65% to 85-98% in multi-person scenes.
- **Qwen3.5 think stripping** — Qwen3.5 outputs empty `<think>` block by default; stripped before JSON parsing.
- **Single-class > Multi-class prompt** — Per-class prompts achieve VLM grounding close to GT bbox (gap ~0.03). Multi-class prompts (listing 20 classes) drop mIoU by ~0.20 due to VLM overload in complex scenes. Recommended mode for practical use: single-category prompts.
- **Multi-class training benefits** — Training mask decoder on all 20 classes (2157 samples) improves person GT ceiling from 0.759 to 0.765 vs single-class training. The mask decoder generalizes better with diverse bbox shapes from different classes.

## File Map

| File | Role |
|------|------|
| `model/qwenvlseg.py` | Main model: frozen LLM + mask decoder (supports Qwen3-VL & Qwen3.5) |
| `model/mask_decoder.py` | Box-guided mask decoder (paper Section 3.2) |
| `data/voc_dataset.py` | VOC 2012 data loader |
| `data/prompts.py` | ChatML prompt templates |
| `train.py` | Training loop: AdamW + cosine, bbox noise |
| `eval.py` | Evaluation: mIoU, cIoU, Dice, P@X |
| `eval_gt_bbox.py` | GT bbox upper bound benchmark |
| `inference.py` | Single-image inference with visualization |
| `visualize.py` | Batch visualization (GT vs Pred comparison) |
| `test_forward.py` | Smoke test |
| `test_qwen35_bbox_format.py` | Qwen3.5 bbox grounding format diagnosis |
| `test_qwen35_thinking.py` | Qwen3.5 thinking mode behavior test |
| `config.yaml` | Configuration |
| `debugnotes.md` | Debugging history with root cause analysis |

## Notes

- LLM uses `self.base_model.generate()` (HF native path) — manual token-by-token loop fails after `resize_token_embeddings`.
- Image input to processor must be PIL Image, not raw tensor.
- Training target format matches the model's native markdown JSON output.
- On AutoDL, set `model_cache_dir` in config to download from ModelScope (much faster than HuggingFace).
