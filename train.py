"""Training script for QwenVLSeg lightweight reproduction."""

import os
import math
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image

from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import VOCSegDataset, resize_and_pad
from data.prompts import build_category_prompt, build_target_json


class TrainingDataset(Dataset):
    """Wraps VOCSegDataset to produce pre-tokenized training samples."""

    def __init__(self, voc_dataset: VOCSegDataset, processor, image_size: int = 512):
        self.voc_dataset = voc_dataset
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image_pil, mask, bbox, category = self.voc_dataset[idx]

        # Build ChatML prompt text
        user_instruction = build_category_prompt(category)
        target_json = build_target_json([bbox], category)

        user_content = f"<|vision_start|><|image_pad|><|vision_end|>\n{user_instruction}"
        assistant_content = target_json

        # Full ChatML with image placeholder
        system_part = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
        user_part = f"<|im_start|>user\n{user_content}<|im_end|>"
        assistant_part = f"<|im_start|>assistant\n{assistant_content}<|im_end|>"

        full_prompt = f"{system_part}\n{user_part}\n{assistant_part}"

        # Tokenize with processor (produces correct pixel_values, image_grid_thw, input_ids)
        inputs = self.processor(
            text=[full_prompt],
            images=[image_pil],
            return_tensors="pt",
            padding=False,
        )

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'image_grid_thw': inputs['image_grid_thw'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'mask': torch.tensor(mask).long(),
            'bbox': torch.tensor(bbox).float(),
            'image_pil': image_pil,
            'full_prompt': full_prompt,
            'user_part': f"{system_part}\n{user_part}",
            'assistant_part': assistant_part,
        }


def collate_fn(batch):
    """Pad and collate training samples."""
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids_list = []
    attention_list = []
    pixel_values_list = []
    grid_thw_list = []
    masks_list = []
    bboxes_list = []
    orig_images_list = []

    for item in batch:
        seq_len_item = item['input_ids'].shape[0]
        pad_len = max_len - seq_len_item

        # Pad input_ids and attention_mask
        input_ids = F.pad(item['input_ids'], (0, pad_len), value=0)
        attention = F.pad(item['attention_mask'], (0, pad_len), value=0)

        input_ids_list.append(input_ids)
        attention_list.append(attention)
        pixel_values_list.append(item['pixel_values'])
        grid_thw_list.append(item['image_grid_thw'])
        masks_list.append(item['mask'])
        bboxes_list.append(item['bbox'])

        # Original image for CNN stem
        img_np = np.array(item['image_pil'].resize((512, 512))).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        orig_images_list.append(img_tensor)

    # Stack what we can
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_list)

    # Concatenate flat pixel_values
    pixel_values = torch.cat(pixel_values_list, dim=0)  # (sum patches, 1536)
    image_grid_thw = torch.stack(grid_thw_list)  # (B, 3)
    gt_mask = torch.stack(masks_list)
    gt_bbox = torch.stack(bboxes_list)
    orig_images = torch.stack(orig_images_list)

    return {
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'gt_mask': gt_mask,
        'gt_bbox': gt_bbox,
        'orig_images': orig_images,
    }


def compute_miou(pred_mask, gt_mask):
    valid = (gt_mask != -100)
    pred = pred_mask[valid]
    gt = gt_mask[valid].clamp(min=0)
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    intersection = (pred * gt).sum().float()
    union = (pred + gt).clamp(max=1).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()


def validate(model, val_loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=batch['pixel_values'].to(device),
                    image_grid_thw=batch['image_grid_thw'].to(device),
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    orig_images=batch['orig_images'].to(device),
                    gt_mask=batch['gt_mask'].to(device),
                    gt_bbox=batch['gt_bbox'].to(device),
                )
            mask_logits = outputs['mask_logits']
            gt = batch['gt_mask']
            for b in range(mask_logits.shape[0]):
                pred_up = F.interpolate(
                    mask_logits[b:b+1], size=gt.shape[-2:], mode='bilinear'
                ).squeeze()
                pred_bin = (torch.sigmoid(pred_up) > 0.5).float()
                ious.append(compute_miou(pred_bin.cpu(), gt[b].cpu()))
    model.train()
    return sum(ious) / len(ious) if ious else 0.0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    train_cfg = config['training']

    # ── Resolve categories (backward compat: fall back to category/category_id) ──
    categories = data_cfg.get('categories', None)
    if categories is None and 'category' in data_cfg:
        categories = [data_cfg['category']]

    # ── Datasets ──
    train_ds = VOCSegDataset(
        root=data_cfg['root'], split='train',
        categories=categories,
        image_size=data_cfg['image_size'], min_mask_pixels=data_cfg['min_mask_pixels'],
    )
    val_ds = VOCSegDataset(
        root=data_cfg['root'], split='val',
        categories=categories,
        image_size=data_cfg['image_size'], min_mask_pixels=data_cfg['min_mask_pixels'],
    )
    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Model ──
    print("Loading model...")
    model = QwenVLSeg(
        model_name=config['model']['name'],
        mask_decoder_cfg=config['mask_decoder'],
    )
    processor = model.processor

    # ── DataLoaders ──
    train_wrapped = TrainingDataset(train_ds, processor, data_cfg['image_size'])
    val_wrapped = TrainingDataset(val_ds, processor, data_cfg['image_size'])

    train_loader = DataLoader(
        train_wrapped, batch_size=train_cfg['batch_size'],
        shuffle=True, collate_fn=collate_fn, num_workers=train_cfg.get('num_workers', 0),
    )
    val_loader = DataLoader(
        val_wrapped, batch_size=train_cfg['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=train_cfg.get('num_workers', 0),
    )

    model.to(device)

    # ── Resume from checkpoint ──
    if args.resume:
        print(f"Resuming from {args.resume} (loading mask_decoder)")
        model.load_checkpoint(args.resume)

    # ── Trainable params ──
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        trainable, lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )
    grad_accum = train_cfg['gradient_accumulation_steps']
    total_steps = len(train_loader) * train_cfg['max_epochs'] // grad_accum
    warmup = train_cfg['lr_warmup_steps']

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    output_dir = Path(config['output']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_interval = config['output']['log_interval']
    best_miou = 0.0
    global_step = 0

    # Log file
    log_path = output_dir / "training_log.csv"
    log_file = open(log_path, 'w')
    log_file.write("epoch,seg_loss,mIoU,best_mIoU\n")
    log_file.flush()

    for epoch in range(train_cfg['max_epochs']):
        model.train()
        epoch_seg = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['max_epochs']}")
        for step, batch in enumerate(pbar):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=batch['pixel_values'].to(device),
                    image_grid_thw=batch['image_grid_thw'].to(device),
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    orig_images=batch['orig_images'].to(device),
                    gt_mask=batch['gt_mask'].to(device),
                    gt_bbox=batch['gt_bbox'].to(device),
                )

            loss = outputs['loss'] / grad_accum
            loss.backward()

            epoch_seg += outputs['seg_loss'].item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % log_interval == 0:
                pbar.set_postfix({
                    'seg': f"{outputs['seg_loss'].item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Handle remaining accumulated gradients
        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_seg = epoch_seg / max(1, len(train_loader))
        print(f"Epoch {epoch+1}: seg={avg_seg:.4f}")

        miou = 0.0
        if len(val_loader) > 0:
            miou = validate(model, val_loader, device)
            print(f"  Val mIoU: {miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                model.save_checkpoint(str(output_dir / "best_checkpoint.pt"))
                print(f"  Saved best (mIoU={miou:.4f})")

        model.save_checkpoint(str(output_dir / "latest_checkpoint.pt"))

        # Write log
        log_file.write(f"{epoch+1},{avg_seg:.4f},{miou:.4f},{best_miou:.4f}\n")
        log_file.flush()

    log_file.close()
    print(f"\nDone. Best mIoU: {best_miou:.4f}")
    print(f"Log saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (loads mask_decoder)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
