"""Visualize: compare GT vs predicted mask on val samples."""
import argparse, os, yaml
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import VOCSegDataset, resize_and_pad, VOC_CLASSES
from data.prompts import build_category_prompt


def visualize_samples(
    model: QwenVLSeg,
    config: dict,
    checkpoint_path: str,
    num_samples: int = 5,
    output_dir: str = "viz_output",
    device: str = "cuda",
):
    model.load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    data_cfg = config['data']
    categories = data_cfg.get('categories', None)
    if categories is None and 'category' in data_cfg:
        categories = [data_cfg['category']]

    ds = VOCSegDataset(
        root=data_cfg['root'], split='val',
        categories=categories,
        image_size=data_cfg['image_size'], min_mask_pixels=data_cfg['min_mask_pixels'],
    )
    os.makedirs(output_dir, exist_ok=True)
    image_size = data_cfg['image_size']
    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)

    for n, idx in enumerate(indices):
        print(f"\nSample {n+1}/{num_samples} (idx={idx})")
        image_pil, gt_mask, gt_bbox, category = ds[idx]

        # Run inference
        image_resized, _, ox, oy, nw, nh = resize_and_pad(
            image_pil, Image.new("L", image_pil.size, 0), image_size
        )
        img_tensor = torch.from_numpy(
            np.array(image_resized).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model.generate_and_segment(
                pixel_values=img_tensor,
                image_grid_thw=torch.tensor([[1, image_size // 16, image_size // 16]], device=device),
                prompt_text=build_category_prompt(category),
                max_new_tokens=512,
                image_size=image_size,
            )

        pred_mask = result['mask']
        if pred_mask is None:
            print(f"  SKIP: bbox parse failed")
            continue

        # Keep only valid region (undo padding)
        pred_cropped = pred_mask[oy:oy+nh, ox:ox+nw].astype(np.float32)
        gt_cropped = gt_mask[oy:oy+nh, ox:ox+nw].astype(np.float32)
        valid = (gt_cropped != -100)
        pred_bin = (pred_cropped > 0.5).astype(np.float32)
        gt_bin = (gt_cropped > 0).astype(np.float32)
        inter = float((pred_bin * gt_bin * valid).sum())
        union = float(((pred_bin + gt_bin > 0) & valid).sum())
        iou = inter / max(union, 1)
        print(f"  Bboxes: {result['bboxes']} | IoU: {iou:.4f}")

        # Build 4 individual images: Input, GT overlay, Prediction overlay, Binary Mask
        img_np = np.array(image_resized)
        h, w = image_size, image_size

        # Pred mask overlay (red)
        pred_overlay = img_np.copy()
        pred_overlay[pred_mask > 0.5] = [255, 0, 0]
        pred_overlay = cv2.addWeighted(img_np, 0.6, pred_overlay, 0.4, 0)

        # GT mask overlay (green)
        gt_vis = np.zeros_like(img_np)
        gt_vis[gt_mask > 0] = [0, 255, 0]
        gt_overlay = cv2.addWeighted(img_np, 0.6, gt_vis, 0.4, 0)

        # Binary mask visualization
        bin_vis = np.dstack([pred_mask.astype(np.uint8)] * 3) * 255

        # Save individual images
        for name, arr in [("input", img_np), ("gt", gt_overlay),
                           ("pred", pred_overlay), ("mask", bin_vis)]:
            out_path = os.path.join(output_dir, f"sample_{idx}_{name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        # Save IoU value
        print(f"  Saved: sample_{idx}_*.png | IoU={iou:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/viz_output")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = QwenVLSeg(
        model_name=config['model']['name'],
        mask_decoder_cfg=config['mask_decoder'],
    )
    visualize_samples(model, config, args.checkpoint, args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()
