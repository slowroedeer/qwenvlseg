# QwenVLSeg — Lightweight Reproduction of Qwen3-VL-Seg

Lightweight reproduction of **Qwen3-VL-Seg** (May 2026). Extends Qwen3-VL's multimodal understanding to pixel-level segmentation by attaching a box-guided mask decoder to the frozen VLM.

- **LLM**: Qwen3-VL-2B-Instruct (completely frozen)
- **Mask Decoder**: ~5M params, full training
- **Data**: Pascal VOC 2012, person class
- **GPU**: RTX 4080 SUPER (32GB)

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

LLM is completely frozen — Qwen3-VL-2B-Instruct's native 2D grounding provides bbox output. The mask decoder is the only trainable component. See `experiment_report.md` for full architecture details.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.yaml` to set your data path.

## Workflow

### 1. Smoke test

```bash
python test_forward.py
```

Verifies model loading, ViT shapes, and forward pass on a synthetic red image.

### 2. Overfitting test

```bash
python debug_train.py
```

Overfits 5 images for 200 steps. Confirms the mask decoder can learn and produces correct overlay visualizations.

### 3. Train

```bash
python train.py --config config.yaml
```

Trains only the mask decoder (~5M params). LLM and ViT are frozen. Training uses noisy GT bbox (±10%) for robustness to LLM bbox errors.

Key config in `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen3-VL-2B-Instruct"
mask_decoder:
  hidden_dim: 256
  num_transformer_layers: 2
data:
  root: "/path/to/VOC2012"
  category: "person"
  category_id: 15
training:
  max_epochs: 20
  batch_size: 4
  learning_rate: 1.0e-4
```

### 4. Evaluate

```bash
# Full val set (440 samples)
python eval.py --checkpoint /path/to/best_checkpoint.pt --config config.yaml --max-samples 0

# GT bbox upper bound
python eval_gt_bbox.py
```

Metrics: mIoU, cIoU, Mean Dice, P@0.5, P@0.7 (aligned with Qwen3-VL-Seg paper).

### 5. Single image inference

```bash
python inference.py \
  --image /path/to/image.jpg \
  --category person \
  --checkpoint /path/to/best_checkpoint.pt \
  --output mask.png \
  --vis overlay.png
```

### 6. Visualization (batch)

```bash
python visualize.py \
  --checkpoint /path/to/best_checkpoint.pt \
  --num-samples 12 \
  --output-dir /root/autodl-tmp/report_viz
```

Generates 4-panel comparison images (Input | GT | Prediction | Binary Mask).

## Results

| Metric | Our Result | GT Bbox Upper Bound |
|--------|-----------|---------------------|
| mIoU | **0.7539** | 0.7975 |
| cIoU | **0.8159** | 0.8592 |
| Mean Dice | **0.8412** | 0.8764 |
| P@0.5 | **0.8864** | 0.9568 |
| P@0.7 | **0.7341** | 0.8455 |
| Bbox parse rate | **100%** | — |

440 val samples. Multi-person scenes benefit from per-instance LLM bboxes + max merge.

## Key Design Decisions

- **LLM completely frozen** — Qwen3-VL-2B-Instruct already has native 2D grounding. LoRA training was found to destroy this capability (see `debugnotes.md`).
- **Two-pass inference** — Pass 1: LLM generates bbox JSON. Pass 2: insert mask tokens, forward LLM to extract h_mask.
- **Bbox noise training** — ±10% perturbation on GT bbox bridges the gap between precise GT and approximate LLM bboxes.
- **Multi-instance max merge** — Per-instance mask decode + pixel-wise max, improving coverage from 65% to 85-98% in multi-person scenes.

## File Map

| File | Role |
|------|------|
| `model/qwenvlseg.py` | Main model: frozen LLM + mask decoder |
| `model/mask_decoder.py` | Box-guided mask decoder (paper Section 3.2) |
| `data/voc_dataset.py` | VOC 2012 data loader |
| `data/prompts.py` | ChatML prompt templates |
| `train.py` | Training loop: AdamW + cosine, bbox noise |
| `eval.py` | Evaluation: mIoU, cIoU, Dice, P@X |
| `eval_gt_bbox.py` | GT bbox upper bound benchmark |
| `inference.py` | Single-image inference with visualization |
| `visualize.py` | Batch visualization (GT vs Pred comparison) |
| `debug_train.py` | Overfitting test |
| `test_forward.py` | Smoke test |
| `config.yaml` | Configuration |
| `experiment_report.md` | Full experiment report with architecture details |
| `debugnotes.md` | Debugging history with root cause analysis |

## Notes

- LLM uses `self.base_model.generate()` (HF native path) — manual token-by-token loop fails after `resize_token_embeddings`.
- Image input to processor must be PIL Image, not raw tensor.
- Training target format matches Qwen3-VL-2B-Instruct's native markdown JSON output: `\n` `` ```json\n[...]\n``` ``.
