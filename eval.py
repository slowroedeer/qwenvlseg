"""Evaluation script — compute mIoU, cIoU, Dice, P@0.5, P@0.7 on VOC val set.

Aligns with Qwen3-VL-Seg paper metrics:
- mIoU: mean of per-sample IoU
- cIoU: cumulative IoU (Σintersection / Σunion over all samples)
- P@0.5 / P@0.7: precision at IoU thresholds
"""

import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import VOCSegDataset, resize_and_pad
from data.prompts import build_category_prompt


def compute_metrics(pred_mask, gt_mask):
    """Compute IoU, Dice, intersection, union for a single sample.

    Returns:
        iou, dice, intersection (int), union (int)
    """
    valid = (gt_mask != -100)
    pred = pred_mask[valid]
    gt = (gt_mask[valid] > 0).astype(np.int64)

    intersection = int((pred * gt).sum())
    union = int(((pred + gt) > 0).sum())

    iou = intersection / max(union, 1)
    dice = 2 * intersection / max(int(pred.sum()) + int(gt.sum()), 1)

    return iou, dice, intersection, union


def evaluate(
    model: QwenVLSeg,
    config: dict,
    checkpoint_path: str,
    device: str = "cuda",
    max_samples: int = 0,
) -> dict:
    """Evaluate on VOC val set."""
    model.load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    data_cfg = config['data']
    val_ds = VOCSegDataset(
        root=data_cfg['root'],
        split='val',
        category=data_cfg['category'],
        category_id=data_cfg['category_id'],
        image_size=data_cfg['image_size'],
        min_mask_pixels=data_cfg['min_mask_pixels'],
    )
    print(f"Evaluating on {len(val_ds)} val samples with person")
    if max_samples > 0:
        val_ds.valid_ids = val_ds.valid_ids[:max_samples]
        print(f"  (limited to {len(val_ds.valid_ids)} samples)")

    category = data_cfg['category']
    image_size = data_cfg['image_size']
    ious = []
    dices = []
    total_intersection = 0
    total_union = 0
    bbox_correct = 0
    total = 0

    for idx in tqdm(range(len(val_ds)), desc="Evaluating"):
        image_pil, gt_mask_np, gt_bbox = val_ds[idx]

        # For inference, use generate_and_segment
        image_resized, _, ox, oy, nw, nh = resize_and_pad(
            image_pil, Image.new("L", image_pil.size, 0), image_size
        )

        prompt = build_category_prompt(category)

        img_tensor = (
            torch.from_numpy(np.array(image_resized))
            .permute(2, 0, 1).float() / 255.0
        ).unsqueeze(0).to(device)

        img_grid_thw = torch.tensor(
            [[1, image_size // 16, image_size // 16]], device=device
        )

        with torch.no_grad():
            result = model.generate_and_segment(
                pixel_values=img_tensor,
                image_grid_thw=img_grid_thw,
                prompt_text=prompt,
                max_new_tokens=512,
                image_size=image_size,
            )

        pred_mask = result['mask']
        if pred_mask is not None:
            iou, dice, inter, uni = compute_metrics(pred_mask, gt_mask_np)
            ious.append(iou)
            dices.append(dice)
            total_intersection += inter
            total_union += uni
        else:
            ious.append(0.0)
            dices.append(0.0)

        if len(result['bboxes']) > 0:
            bbox_correct += 1
        total += 1

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    c_iou = total_intersection / max(total_union, 1)
    prec_50 = np.mean([1.0 if i >= 0.5 else 0.0 for i in ious])
    prec_70 = np.mean([1.0 if i >= 0.7 else 0.0 for i in ious])

    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {total}")
    print(f"mIoU:      {mean_iou:.4f}")
    print(f"cIoU:      {c_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"P@0.5:     {prec_50:.4f}")
    print(f"P@0.7:     {prec_70:.4f}")
    print(f"Bbox parse rate: {bbox_correct}/{total} ({100*bbox_correct/total:.1f}%)")

    return {
        'mIoU': mean_iou,
        'cIoU': c_iou,
        'dice': mean_dice,
        'prec@0.5': prec_50,
        'prec@0.7': prec_70,
        'bbox_parse_rate': bbox_correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate QwenVLSeg")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit eval to first N samples (0 = all)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = QwenVLSeg(
        model_name=config['model']['name'],
        mask_decoder_cfg=config['mask_decoder'],
    )

    metrics = evaluate(model, config, args.checkpoint, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
