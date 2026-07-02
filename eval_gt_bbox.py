"""Evaluate mask decoder upper bound using GT bbox on full val set — multi-class."""
import argparse, os
import torch, yaml, numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import VOCSegDataset, resize_and_pad, VOC_CLASSES, mask_to_bbox
from data.prompts import build_target_json, build_category_prompt
from eval import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = QwenVLSeg(
        model_name=config['model']['name'],
        mask_decoder_cfg=config['mask_decoder'],
    )
    model.load_checkpoint(args.checkpoint)
    model.cuda().eval()
    model.base_model = model.base_model.to(dtype=torch.bfloat16)

    data_cfg = config['data']
    image_size = data_cfg['image_size']

    # Resolve categories
    categories = data_cfg.get('categories', None)
    if categories is None and 'category' in data_cfg:
        categories = [data_cfg['category']]
    elif categories == "all":
        categories = VOC_CLASSES[1:]
    elif isinstance(categories, str):
        categories = [categories]

    name_to_id = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    # Load val image IDs
    split_file = os.path.join(data_cfg['root'], "ImageSets", "Segmentation", "val.txt")
    with open(split_file) as f:
        val_ids = [line.strip() for line in f if line.strip()]

    # Per-class accumulators
    class_metrics = {
        cat: {"ious": [], "dices": [], "inter": 0, "union": 0}
        for cat in categories
    }
    all_ious = []
    total_inter = 0
    total_union = 0

    for img_id in tqdm(val_ids, desc="GT bbox eval"):
        img_path = os.path.join(data_cfg['root'], "JPEGImages", f"{img_id}.jpg")
        mask_path = os.path.join(data_cfg['root'], "SegmentationClass", f"{img_id}.png")

        image_pil = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))

        # Resize image once (same for all classes)
        img_resized, _, ox, oy, nw, nh = resize_and_pad(
            image_pil, Image.new("L", image_pil.size, 0), image_size
        )
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

        for cat_name in categories:
            cat_id = name_to_id[cat_name]

            # Skip if class not present in this image
            if (raw_mask == cat_id).sum() < data_cfg['min_mask_pixels']:
                continue

            # Binary mask and resize (image already resized, only mask matters here)
            binary_mask = np.zeros_like(raw_mask, dtype=np.uint8)
            binary_mask[raw_mask == cat_id] = 1
            binary_mask[raw_mask == 255] = 255
            binary_mask_img = Image.fromarray(binary_mask, "L")

            # Resize mask only, reuse resized image
            _, mask_img, _, _, _, _ = resize_and_pad(
                image_pil, binary_mask_img, image_size
            )
            gt_mask = np.array(mask_img).astype(np.int64)
            gt_mask[gt_mask == 255] = -100

            # GT bbox
            binary_bbox = (gt_mask == 1).astype(np.uint8)
            gt_bbox = mask_to_bbox(binary_bbox, image_size, image_size)
            if binary_bbox.sum() == 0:
                gt_bbox = [0, 0, 1000, 1000]

            # Build prompt with GT bbox + mask tokens
            target_json = build_target_json([gt_bbox], cat_name)
            user_instruction = build_category_prompt(cat_name)
            user_content = f'<|vision_start|><|image_pad|><|vision_end|>\n{user_instruction}'
            system_part = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
            user_part = f'<|im_start|>user\n{user_content}<|im_end|>'
            assistant_part = f'<|im_start|>assistant\n{target_json}<|im_end|>'
            full_prompt = f'{system_part}\n{user_part}\n{assistant_part}'

            inputs = model.processor(text=[full_prompt], images=[img_resized], return_tensors='pt')
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.no_grad():
                out = model(
                    pixel_values=inputs['pixel_values'],
                    image_grid_thw=inputs['image_grid_thw'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    orig_images=img_t,
                    gt_mask=torch.tensor(gt_mask).unsqueeze(0).long().cuda(),
                    gt_bbox=torch.tensor(gt_bbox).unsqueeze(0).float().cuda(),
                )

            mask_up = F.interpolate(
                out['mask_logits'], size=gt_mask.shape, mode='bilinear'
            )
            pred = (torch.sigmoid(mask_up.squeeze()) > 0.5).float().cpu().numpy()
            iou, dice, inter, uni = compute_metrics(pred, gt_mask)

            class_metrics[cat_name]["ious"].append(iou)
            class_metrics[cat_name]["dices"].append(dice)
            class_metrics[cat_name]["inter"] += inter
            class_metrics[cat_name]["union"] += uni
            all_ious.append(iou)
            total_inter += inter
            total_union += uni

    # Report
    print(f"\n=== GT Bbox Upper Bound ({len(val_ids)} val images) ===")
    print()
    per_class_miou = {}
    for cat in categories:
        m = class_metrics[cat]
        n = len(m["ious"])
        miou = np.mean(m["ious"]) if m["ious"] else 0.0
        ciou = m["inter"] / max(m["union"], 1)
        per_class_miou[cat] = miou
        print(f"  {cat:15s}  mIoU={miou:.4f}  cIoU={ciou:.4f}  n={n}")

    overall_miou = np.mean(list(per_class_miou.values()))
    overall_ciou = total_inter / max(total_union, 1)
    mean_dice = np.mean([np.mean(m["dices"]) for m in class_metrics.values() if m["dices"]]) if all_ious else 0.0
    prec_50 = np.mean([1.0 if i >= 0.5 else 0.0 for i in all_ious]) if all_ious else 0.0
    prec_70 = np.mean([1.0 if i >= 0.7 else 0.0 for i in all_ious]) if all_ious else 0.0

    print(f"\n  {'MEAN':15s}  mIoU={overall_miou:.4f}  cIoU={overall_ciou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  P@0.5:     {prec_50:.4f}")
    print(f"  P@0.7:     {prec_70:.4f}")


if __name__ == "__main__":
    main()
