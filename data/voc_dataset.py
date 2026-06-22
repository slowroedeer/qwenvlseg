"""Pascal VOC 2012 dataset — converts to instruction segmentation format."""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]


def resize_and_pad(image, mask, target_size=512):
    """Resize image and mask to target_size × target_size, keeping aspect ratio with padding."""
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    image = image.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)

    # Pad to target_size
    canvas_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas_mask = Image.new("L", (target_size, target_size), 255)  # 255 = ignore

    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2

    canvas_img.paste(image, (offset_x, offset_y))
    canvas_mask.paste(mask, (offset_x, offset_y))

    return canvas_img, canvas_mask, offset_x, offset_y, new_w, new_h


def mask_to_bbox(binary_mask, img_w, img_h, coord_scale=1000):
    """Convert binary mask to bounding box in [0, coord_scale] normalized coords."""
    if binary_mask.sum() == 0:
        return [0, 0, coord_scale, coord_scale]

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, coord_scale, coord_scale]

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    x1 = int(x1 / img_w * coord_scale)
    y1 = int(y1 / img_h * coord_scale)
    x2 = int(x2 / img_w * coord_scale)
    y2 = int(y2 / img_h * coord_scale)

    return [x1, y1, x2, y2]


class VOCSegDataset(Dataset):
    """Pascal VOC 2012 dataset for instruction-based single-class segmentation.

    Each sample returns:
        image: PIL Image (RGB)
        mask:  torch.Tensor [H, W], 0=bg, 1=target_class, -100=ignore
        bbox:  list [x1,y1,x2,y2] in 0-1000 coords
        prompt_text: str — full user prompt (ChatML format)
        target_text: str — assistant target output (JSON)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        category: str = "person",
        category_id: int = 15,
        image_size: int = 512,
        min_mask_pixels: int = 100,
    ):
        self.root = root
        self.split = split
        self.category = category
        self.category_id = category_id
        self.image_size = image_size
        self.min_mask_pixels = min_mask_pixels

        # Load image list
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # Filter: only keep images containing the target category
        self.valid_ids = []
        for img_id in self.ids:
            mask_path = os.path.join(root, "SegmentationClass", f"{img_id}.png")
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                if (mask == self.category_id).sum() >= self.min_mask_pixels:
                    self.valid_ids.append(img_id)

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]

        img_path = os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")
        mask_path = os.path.join(self.root, "SegmentationClass", f"{img_id}.png")

        image = Image.open(img_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))

        # Binary mask: 1=person, 0=other, 255=void (keep as uint8 for PIL compatibility)
        binary_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        binary_mask[raw_mask == self.category_id] = 1
        binary_mask[raw_mask == 255] = 255  # void → 255 (PIL-safe)

        # Resize with padding — mask stays uint8 {0, 1, 255} throughout PIL processing
        orig_w, orig_h = image.size
        binary_mask_img = Image.fromarray(binary_mask, mode="L")

        image, mask_img, ox, oy, new_w, new_h = resize_and_pad(
            image, binary_mask_img, self.image_size
        )

        # Convert to int64 and map void 255 → -100 for loss computation
        mask = np.array(mask_img).astype(np.int64)
        mask[mask == 255] = -100

        # Compute bbox from binary mask (exclude ignore pixels)
        binary_mask_bbox = (mask == 1).astype(np.uint8)
        bbox = mask_to_bbox(binary_mask_bbox, self.image_size, self.image_size)

        # If no valid person pixels after resize, use dummy bbox
        if binary_mask_bbox.sum() == 0:
            bbox = [0, 0, 1000, 1000]

        return image, mask, bbox
