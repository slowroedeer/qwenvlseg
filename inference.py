"""Inference script — input image + instruction → segmentation mask."""

import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import resize_and_pad
from data.prompts import build_category_prompt


def inference(
    model: QwenVLSeg,
    image_path: str,
    category: str = "person",
    checkpoint_path: str | None = None,
    image_size: int = 512,
    device: str = "cuda",
) -> dict:
    """Run inference on a single image."""
    if checkpoint_path is not None:
        model.load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # Load and preprocess image
    image_pil = Image.open(image_path).convert("RGB")
    image_pil = image_pil.resize((image_size, image_size), Image.BILINEAR)

    img_tensor = torch.from_numpy(
        np.array(image_pil).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    prompt_text = build_category_prompt(category)

    with torch.no_grad():
        result = model.generate_and_segment(
            pixel_values=img_tensor,
            prompt_text=prompt_text,
            image_grid_thw=torch.tensor([[1, image_size//16, image_size//16]], device=device),
            max_new_tokens=256,
            image_size=image_size,
        )

    print(f"Text output: {result['text_output']}")
    print(f"Parsed bboxes: {result['bboxes']}")
    print(f"IoU score: {result['iou_score']:.4f}")
    return result


def visualize(image_path, mask, output_path):
    import cv2
    image = cv2.imread(image_path)
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    overlay = image.copy()
    overlay[mask > 0.5] = [0, 0, 255]
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(output_path, result)
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--category", type=str, default="person")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default="output_mask.png")
    parser.add_argument("--vis", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = QwenVLSeg(
        model_name=config['model']['name'],
        mask_decoder_cfg=config['mask_decoder'],
    )

    result = inference(
        model, args.image, args.category, args.checkpoint,
        image_size=config['data']['image_size'],
    )

    mask_img = (result['mask'] * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(args.output)
    print(f"Mask saved to {args.output}")

    if args.vis:
        visualize(args.image, result['mask'], args.vis)


if __name__ == "__main__":
    main()
