"""Generate 20-class per-class visualization examples on VOC val set."""
import argparse, os
import numpy as np
from PIL import Image
import torch
import cv2

from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import resize_and_pad, VOC_CLASSES
from data.prompts import build_category_prompt


def compute_iou(pred_mask, gt_binary):
    inter = float((pred_mask * gt_binary).sum())
    union = float(((pred_mask + gt_binary) > 0).sum())
    return inter / max(union, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="VOC2012 root path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="0.8B checkpoint path")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda'
    image_size = args.image_size

    mask_cfg = {"hidden_dim": 256, "num_transformer_layers": 2, "num_heads": 8,
                "num_queries": 1, "mask_stride": 4}

    print("Loading model...")
    model = QwenVLSeg(model_name="Qwen/Qwen3.5-0.8B", mask_decoder_cfg=mask_cfg)
    model.load_checkpoint(args.checkpoint)
    model.to(device)
    model.base_model = model.base_model.to(dtype=torch.bfloat16)
    model.eval()

    name_to_id = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    # Load val image IDs
    split_file = os.path.join(args.data_root, "ImageSets", "Segmentation", "val.txt")
    with open(split_file) as f:
        val_ids = [line.strip() for line in f if line.strip()]

    # Pre-compute which images contain each class
    class_images = {cat: [] for cat in VOC_CLASSES[1:]}
    for img_id in val_ids:
        mask_path = os.path.join(args.data_root, "SegmentationClass", f"{img_id}.png")
        if not os.path.exists(mask_path):
            continue
        raw_mask = np.array(Image.open(mask_path))
        for cat_name in VOC_CLASSES[1:]:
            cat_id = name_to_id[cat_name]
            if (raw_mask == cat_id).sum() >= 100:
                class_images[cat_name].append(img_id)

    # Pick classes to visualize (good performers from GT eval)
    viz_classes = [
        "cat", "bird", "cow", "train", "dog", "bus",
        "sheep", "horse", "motorbike", "boat", "aeroplane",
        "person", "car", "sofa", "bottle", "tvmonitor",
        "bicycle", "chair", "diningtable", "pottedplant",
    ]

    for cat_name in viz_classes:
        img_ids = class_images.get(cat_name, [])
        if not img_ids:
            print(f"  {cat_name}: no images found")
            continue

        img_id = img_ids[0]  # pick first image
        print(f"{cat_name}: {img_id}")

        img_path = os.path.join(args.data_root, "JPEGImages", f"{img_id}.jpg")
        mask_path = os.path.join(args.data_root, "SegmentationClass", f"{img_id}.png")

        image_pil = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))
        cat_id = name_to_id[cat_name]

        # Resize with padding
        image_resized, _, ox, oy, nw, nh = resize_and_pad(
            image_pil, Image.new("L", image_pil.size, 0), image_size
        )

        # GT binary mask (resized same as image)
        gt_binary = (raw_mask == cat_id).astype(np.uint8)
        gt_pil = Image.fromarray(gt_binary, "L").resize((image_size, image_size), Image.NEAREST)
        gt_resized = np.array(gt_pil)

        # GT overlay
        img_np = np.array(image_resized)
        gt_vis = np.zeros_like(img_np)
        gt_vis[gt_resized > 0] = [0, 255, 0]
        gt_overlay = cv2.addWeighted(img_np, 0.6, gt_vis, 0.4, 0)

        prefix = f"{cat_name}"
        cv2.imwrite(os.path.join(args.output_dir, f"{prefix}_input.png"),
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output_dir, f"{prefix}_gt.png"),
                    cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))

        # Run inference with per-class prompt
        img_tensor = torch.from_numpy(
            img_np.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model.generate_and_segment(
                pixel_values=img_tensor,
                image_grid_thw=torch.tensor([[1, image_size//16, image_size//16]], device=device),
                prompt_text=build_category_prompt(cat_name),
                max_new_tokens=256,
                image_size=image_size,
            )

        pred_mask = result['mask']
        if pred_mask is None:
            print(f"  SKIP: parse failed")
            continue

        # Pred overlay (red)
        pred_overlay = img_np.copy()
        pred_overlay[pred_mask > 0.5] = [255, 0, 0]
        pred_overlay = cv2.addWeighted(img_np, 0.6, pred_overlay, 0.4, 0)

        # Binary mask
        bin_vis = np.dstack([pred_mask.astype(np.uint8)] * 3) * 255

        # IoU
        iou = compute_iou(pred_mask, gt_resized)
        print(f"  bboxes={result['bboxes']} IoU={iou:.4f}")

        cv2.imwrite(os.path.join(args.output_dir, f"{prefix}_pred.png"),
                    cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output_dir, f"{prefix}_mask.png"),
                    cv2.cvtColor(bin_vis, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {prefix}_*.png | IoU={iou:.4f}")

    print(f"\nDone. All images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
