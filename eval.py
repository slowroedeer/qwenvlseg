"""Evaluation script — compute mIoU, cIoU, Dice, P@0.5, P@0.7 on VOC val set.

Aligns with Qwen3-VL-Seg paper metrics:
- mIoU: mean of per-sample IoU
- cIoU: cumulative IoU (Σintersection / Σunion over all samples)
- P@0.5 / P@0.7: precision at IoU thresholds

Supports both single-class (backward compat) and multi-class evaluation.
"""

import argparse, os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import VOCSegDataset, resize_and_pad, VOC_CLASSES
from data.prompts import build_category_prompt, build_multi_category_prompt

# Normalize LLM label output to VOC class names
LABEL_NORMALIZE = {
    "dining table": "diningtable",
    "potted plant": "pottedplant",
    "tv monitor": "tvmonitor",
    "aero plane": "aeroplane",
    "motor bike": "motorbike",
    "sofa": "sofa",
}


def _normalize_label(label: str) -> str:
    label = label.strip().lower()
    return LABEL_NORMALIZE.get(label, label)


def compute_metrics(pred_mask, gt_mask):
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
    model.load_checkpoint(checkpoint_path)
    model.to(device)
    model.base_model = model.base_model.to(dtype=torch.bfloat16)
    model.eval()

    data_cfg = config['data']
    image_size = data_cfg['image_size']

    # Resolve categories
    categories = data_cfg.get('categories', None)
    if categories is None and 'category' in data_cfg:
        categories = [data_cfg['category']]
    elif categories == "all":
        categories = VOC_CLASSES[1:]  # exclude background
    elif isinstance(categories, str):
        categories = [categories]

    name_to_id = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    # Load val image IDs
    split_file = os.path.join(data_cfg['root'], "ImageSets", "Segmentation", "val.txt")
    with open(split_file, "r") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    if max_samples > 0:
        val_ids = val_ids[:max_samples]

    # Single-class mode
    if len(categories) == 1:
        return _evaluate_single_class(
            model, data_cfg, categories[0], name_to_id[categories[0]],
            val_ids, image_size, device
        )

    # Multi-class mode
    prompt_text = build_multi_category_prompt(categories)
    print(f"Evaluating {len(categories)} classes on {len(val_ids)} val images (multi-class prompt)")

    # Per-class metric accumulators
    class_metrics = {
        cat: {"ious": [], "dices": [], "inter": 0, "union": 0}
        for cat in categories
    }
    all_ious = []
    total_intersection = 0
    total_union = 0
    bbox_correct = 0
    total_images = 0

    for img_id in tqdm(val_ids, desc="Evaluating (multi-class)"):
        img_path = os.path.join(data_cfg['root'], "JPEGImages", f"{img_id}.jpg")
        mask_path = os.path.join(data_cfg['root'], "SegmentationClass", f"{img_id}.png")

        image_pil = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))  # values 0-20, 255=void

        # Resize image + full mask with same padding
        raw_mask_img = Image.fromarray(raw_mask.astype(np.uint8), "L")
        image_resized, mask_resized_img, ox, oy, nw, nh = resize_and_pad(
            image_pil, raw_mask_img, image_size
        )
        resized_mask = np.array(mask_resized_img).astype(np.int64)

        # Model input
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
                prompt_text=prompt_text,
                max_new_tokens=256,
                image_size=image_size,
            )

        per_class_preds = result.get('per_class_masks', {})
        total_images += 1
        if len(result['bboxes']) > 0:
            bbox_correct += 1

        # Normalize predicted labels
        normalized_preds = {}
        for label, mask in per_class_preds.items():
            norm = _normalize_label(label)
            if norm not in normalized_preds:
                normalized_preds[norm] = mask
            else:
                # Merge same-label masks via max
                normalized_preds[norm] = np.maximum(normalized_preds[norm], mask)

        # Compute per-class metrics
        for cat_name in categories:
            cat_id = name_to_id[cat_name]

            # GT binary mask for this class
            gt_binary = np.zeros_like(resized_mask, dtype=np.int64)
            gt_binary[resized_mask == cat_id] = 1
            gt_binary[resized_mask == 255] = -100  # void → ignore

            # Skip if no GT pixels for this class in this image
            if (gt_binary == 1).sum() == 0:
                continue

            # Prediction for this class (or all-zeros if not detected)
            pred_binary = normalized_preds.get(cat_name, np.zeros_like(gt_binary))

            iou, dice, inter, union = compute_metrics(pred_binary, gt_binary)

            class_metrics[cat_name]["ious"].append(iou)
            class_metrics[cat_name]["dices"].append(dice)
            class_metrics[cat_name]["inter"] += inter
            class_metrics[cat_name]["union"] += union
            all_ious.append(iou)
            total_intersection += inter
            total_union += union

    # Report
    print(f"\n=== Multi-Class Evaluation Results ===")
    print(f"Total images: {total_images}")
    print(f"Bbox parse rate: {bbox_correct}/{total_images} ({100*bbox_correct/max(total_images,1):.1f}%)")
    print()

    per_class_miou = {}
    per_class_ciou = {}
    for cat in categories:
        m = class_metrics[cat]
        n = len(m["ious"])
        miou = np.mean(m["ious"]) if m["ious"] else 0.0
        ciou = m["inter"] / max(m["union"], 1)
        per_class_miou[cat] = miou
        per_class_ciou[cat] = ciou
        print(f"  {cat:15s}  mIoU={miou:.4f}  cIoU={ciou:.4f}  n={n}")

    overall_miou = np.mean(list(per_class_miou.values()))
    overall_ciou = total_intersection / max(total_union, 1)
    mean_dice = np.mean([np.mean(m["dices"]) for m in class_metrics.values() if m["dices"]]) if all_ious else 0.0
    prec_50 = np.mean([1.0 if i >= 0.5 else 0.0 for i in all_ious]) if all_ious else 0.0
    prec_70 = np.mean([1.0 if i >= 0.7 else 0.0 for i in all_ious]) if all_ious else 0.0

    print(f"\n  {'MEAN':15s}  mIoU={overall_miou:.4f}  cIoU={overall_ciou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  P@0.5:     {prec_50:.4f}")
    print(f"  P@0.7:     {prec_70:.4f}")

    return {
        'per_class_mIoU': per_class_miou,
        'per_class_cIoU': per_class_ciou,
        'mIoU': overall_miou,
        'cIoU': overall_ciou,
        'dice': mean_dice,
        'prec@0.5': prec_50,
        'prec@0.7': prec_70,
        'bbox_parse_rate': bbox_correct / max(total_images, 1),
    }


def _evaluate_single_class(
    model, data_cfg, category, category_id, val_ids, image_size, device
):
    """Single-class evaluation with per-class prompt (matches training)."""
    ds = VOCSegDataset(
        root=data_cfg['root'], split='val',
        categories=[category],
        image_size=image_size, min_mask_pixels=data_cfg['min_mask_pixels'],
    )
    print(f"Evaluating on {len(ds)} val samples with {category}")

    ious = []
    dices = []
    total_intersection = 0
    total_union = 0
    bbox_correct = 0
    total = 0

    for idx in tqdm(range(len(ds)), desc="Evaluating"):
        image_pil, gt_mask, _, _ = ds[idx]

        # Resize image to match dataset padding
        image_resized, _, _, _, _, _ = resize_and_pad(
            image_pil, Image.new("L", image_pil.size, 0), image_size
        )
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
                prompt_text=build_category_prompt(category),
                max_new_tokens=256,
                image_size=image_size,
            )

        pred_mask = result['mask']
        if pred_mask is not None:
            iou, dice, inter, uni = compute_metrics(pred_mask, gt_mask)
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

    mean_iou = np.mean(ious) if ious else 0.0
    mean_dice = np.mean(dices) if dices else 0.0
    c_iou = total_intersection / max(total_union, 1)
    prec_50 = np.mean([1.0 if i >= 0.5 else 0.0 for i in ious]) if ious else 0.0
    prec_70 = np.mean([1.0 if i >= 0.7 else 0.0 for i in ious]) if ious else 0.0

    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {total}")
    print(f"mIoU:      {mean_iou:.4f}")
    print(f"cIoU:      {c_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"P@0.5:     {prec_50:.4f}")
    print(f"P@0.7:     {prec_70:.4f}")
    print(f"Bbox parse rate: {bbox_correct}/{total} ({100*bbox_correct/max(total,1):.1f}%)")

    return {
        'mIoU': mean_iou,
        'cIoU': c_iou,
        'dice': mean_dice,
        'prec@0.5': prec_50,
        'prec@0.7': prec_70,
        'bbox_parse_rate': bbox_correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate QwenVLSeg")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit eval to first N images (0 = all)")
    parser.add_argument("--category", type=str, default=None,
                        help="Override: evaluate single category (backward compat)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override categories if --category is specified
    if args.category:
        config['data']['categories'] = [args.category]

    model = QwenVLSeg(
        model_name=config['model']['name'],
        mask_decoder_cfg=config['mask_decoder'],
    )

    metrics = evaluate(model, config, args.checkpoint, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
