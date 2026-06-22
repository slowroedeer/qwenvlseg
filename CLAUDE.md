# CLAUDE.md — QwenVLSeg 轻量化复现

## User Preferences

**Before modifying any code**, ask for confirmation first. Do not edit, write, or delete files without explicit approval. Research, reading, and analysis are fine without asking.

## Project Overview

Lightweight reproduction of **Qwen3-VL-Seg** paper (May 2026). Extends Qwen3-VL to pixel-level segmentation by attaching a box-guided mask decoder.

- **LLM**: Qwen3-VL-2B-Instruct (completely frozen)
- **Mask Decoder**: ~5M params, full training
- **Data**: Pascal VOC 2012, person class only
- **Hardware**: RTX 4080 SUPER (32GB), AutoDL cloud

## Architecture

```
Image → ViT (frozen) → multi-scale features [layer 1/3, 2/3, pooler_output]
                              ↓
Prompt → LLM (frozen) → native JSON {"bbox_2d": [...], "label": "person"}
                              ↓
                    parse bbox → insert mask tokens → LLM forward → h_mask
                              ↓                          ↓
                         Mask Decoder (box-guided) → binary mask
```

LLM is **completely frozen** — Qwen3-VL-Instruct has native 2D grounding from instruction tuning (no LoRA, no text training). Training uses noisy GT bbox (±10%) for robustness.

## Commands

```bash
# Install
pip install -r requirements.txt

# Smoke test (model loading + shape check)
python test_forward.py

# Overfitting test (5 images, 200 steps, saves visualizations)
python debug_train.py

# Full training
python train.py --config config.yaml

# Resume from checkpoint
python train.py --resume /path/to/checkpoint.pt

# Evaluate (mIoU, cIoU, Dice, P@0.5, P@0.7)
python eval.py --checkpoint /path/to/best_checkpoint.pt --config config.yaml --max-samples 0

# GT bbox upper bound benchmark
python eval_gt_bbox.py

# Single image inference
python inference.py --image /path/to/img.jpg --category person --checkpoint /path/to/checkpoint.pt --vis result.png
```

## File Map

| File | Role |
|------|------|
| `model/qwenvlseg.py` | Main model: loads Qwen3-VL, adds mask tokens, hooks ViT, integrates Mask Decoder |
| `model/mask_decoder.py` | Box-guided mask decoder (paper Section 3.2) |
| `data/voc_dataset.py` | VOC 2012 loader, filters person samples, computes GT bbox from mask |
| `data/prompts.py` | ChatML templates with JSON bbox + mask token target |
| `train.py` | Training loop: AdamW + cosine schedule, bbox noise, checkpoint save/load |
| `eval.py` | Evaluation: mIoU, cIoU, Dice, P@0.5, P@0.7 |
| `inference.py` | Single image inference with optional overlay visualization |
| `config.yaml` | Model name, mask decoder architecture, data path, training hyperparams |
| `debugnotes.md` | Debugging history and root cause analysis |

## Key Design Decisions

**LLM frozen (no LoRA)**: Qwen3-VL-2B-Instruct already has 2D grounding. LoRA training was found to destroy this native capability (see debugnotes.md, Problem 5).

**Two-pass inference**: Pass 1 — LLM generates native bbox JSON. Pass 2 — mask tokens inserted into generated text, LLM forward for h_mask extraction.

**Bbox noise training**: GT bbox perturbed ±10% during training to bridge the gap between precise GT coordinates and approximate LLM outputs (see debugnotes.md, Problem 6).

**HF generate for inference**: Manual token-by-token generation loop fails after `resize_token_embeddings`. Use `self.base_model.generate()` instead (see test_gen_diag.py investigation).

## Environment

- **AutoDL**: Python 3.10, PyTorch 2.8.0, CUDA 12.8
- **Data path**: `/root/autodl-tmp/data/VOCdevkit/VOC2012`
- **Checkpoint dir**: `/root/autodl-tmp/mycheckpoints`
- **GPU**: RTX 4080 SUPER (32GB)

## History

Originally attempted Qwen3.5-0.8B (failed: thinking mechanism). Migrated to Qwen3-VL-2B-Instruct. Tried LoRA training (failed: destroyed native grounding). Current approach: fully frozen LLM + noisy bbox training, achieving mIoU 0.72 (GT ceiling 0.75).
