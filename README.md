# QwenVLSeg — Lightweight Reproduction of Qwen3-VL-Seg

Lightweight reproduction of **Qwen3-VL-Seg** (May 2026). Extends Qwen3-VL's multimodal understanding to pixel-level segmentation by attaching a box-guided mask decoder to the frozen VLM.

- **LLM**: Qwen3.5-0.8B (completely frozen)
- **Mask Decoder**: ~11.6M params, full training
- **Data**: Pascal VOC 2012, person class
- **GPU**: RTX 4080 SUPER (32GB)

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
  category: "person"
  category_id: 15
training:
  max_epochs: 20
  batch_size: 4
  learning_rate: 1.0e-4
```

### 3. Evaluate

```bash
# Full val set (440 samples)
python eval.py --checkpoint /path/to/best_checkpoint.pt --config config.yaml --max-samples 0

# GT bbox upper bound
python eval_gt_bbox.py
```

Metrics: mIoU, cIoU, Mean Dice, P@0.5, P@0.7 (aligned with Qwen3-VL-Seg paper).

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

### Qwen3.5-0.8B (person class, 20 epochs)

| Metric | LLM Bbox | GT Bbox Upper Bound |
|--------|----------|---------------------|
| mIoU | **0.7343** | 0.7588 |
| cIoU | **0.8141** | 0.8300 |
| Mean Dice | **0.8267** | 0.8491 |
| P@0.5 | **0.9000** | 0.9409 |
| P@0.7 | **0.7091** | 0.7273 |
| Bbox parse rate | **100%** | — |

440 val samples. The GT bbox gap (0.024 mIoU) shows Qwen3.5's grounding is highly precise — the bottleneck is the smaller ViT in the 0.8B model, not bbox quality.

### Qwen3-VL-2B-Instruct (person class, 20 epochs, previous run)

| Metric | LLM Bbox | GT Bbox Upper Bound |
|--------|----------|---------------------|
| mIoU | **0.7539** | 0.7975 |
| cIoU | **0.8159** | 0.8592 |
| Mean Dice | **0.8412** | 0.8764 |
| P@0.5 | **0.8864** | 0.9568 |
| P@0.7 | **0.7341** | 0.8455 |

The 0.8B model achieves 97% of the 2B model's mIoU at 40% of the LLM parameters. Trade-off: smaller ViT limits mask quality (GT ceiling 0.759 vs 0.798), but grounding accuracy is comparable.

## Key Design Decisions

- **LLM completely frozen** — Qwen3-VL-2B and Qwen3.5 both have native 2D grounding from instruction tuning. LoRA training was found to destroy this capability (see `debugnotes.md`).
- **Two-pass inference** — Pass 1: LLM generates bbox JSON. Pass 2: insert mask tokens, forward LLM to extract h_mask.
- **Bbox noise training** — ±10% perturbation on GT bbox bridges the gap between precise GT and approximate LLM bboxes.
- **Multi-instance max merge** — Per-instance mask decode + pixel-wise max, improving coverage from 65% to 85-98% in multi-person scenes.
- **Qwen3.5 think stripping** — Qwen3.5 outputs empty `<think>` block by default; stripped before JSON parsing.

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
