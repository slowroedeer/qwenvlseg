"""Evaluate mask decoder upper bound using GT bbox on full val set."""
import torch, yaml, numpy as np
from PIL import Image
from tqdm import tqdm
from model.qwenvlseg import QwenVLSeg
from data.voc_dataset import VOCSegDataset
from eval import compute_metrics

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = QwenVLSeg(
    model_name=config['model']['name'],
    mask_decoder_cfg=config['mask_decoder'],
)
model.load_checkpoint('/root/autodl-tmp/mycheckpoints/best_checkpoint.pt')
model.cuda().eval()

data_cfg = config['data']
image_size = data_cfg['image_size']

ds = VOCSegDataset(
    root=data_cfg['root'], split='val',
    category=data_cfg['category'], category_id=data_cfg['category_id'],
    image_size=image_size, min_mask_pixels=data_cfg['min_mask_pixels'],
)
print(f"Evaluating GT bbox upper bound on {len(ds)} val samples")

ious = []
dices = []
total_inter = 0
total_union = 0

for idx in tqdm(range(len(ds)), desc="GT bbox eval"):
    img_pil, gt_mask, gt_bbox = ds[idx]

    # Build prompt with GT bbox + mask tokens (matches training target format)
    target_json = (
        '\n```json\n[\n\t{"bbox_2d": ['
        f'{gt_bbox[0]}, {gt_bbox[1]}, {gt_bbox[2]}, {gt_bbox[3]}'
        '], "label": "' + data_cfg['category'] + '", "mask": "<mask_start><mask_token><mask_end>"}\n]\n```'
    )

    user_instruction = (
        f'Locate and segment every instance that belongs to the following categories '
        f'"{data_cfg["category"]}", report bbox coordinates and masks in JSON format.'
    )
    user_content = f'<|vision_start|><|image_pad|><|vision_end|>\n{user_instruction}'
    system_part = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
    user_part = f'<|im_start|>user\n{user_content}<|im_end|>'
    assistant_part = f'<|im_start|>assistant\n{target_json}<|im_end|>'
    full_prompt = f'{system_part}\n{user_part}\n{assistant_part}'

    inputs = model.processor(text=[full_prompt], images=[img_pil], return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    img_np = np.array(img_pil.resize((image_size, image_size))).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

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

    mask_up = torch.nn.functional.interpolate(
        out['mask_logits'], size=gt_mask.shape, mode='bilinear'
    )
    pred = (torch.sigmoid(mask_up.squeeze()) > 0.5).float().cpu().numpy()
    iou, dice, inter, uni = compute_metrics(pred, gt_mask)
    ious.append(iou)
    dices.append(dice)
    total_inter += inter
    total_union += uni

mIoU = np.mean(ious)
cIoU = total_inter / max(total_union, 1)
mean_dice = np.mean(dices)
prec_50 = np.mean([1.0 if i >= 0.5 else 0.0 for i in ious])
prec_70 = np.mean([1.0 if i >= 0.7 else 0.0 for i in ious])

print(f"\n=== GT Bbox Upper Bound ({len(ds)} samples) ===")
print(f"mIoU:      {mIoU:.4f}")
print(f"cIoU:      {cIoU:.4f}")
print(f"Mean Dice: {mean_dice:.4f}")
print(f"P@0.5:     {prec_50:.4f}")
print(f"P@0.7:     {prec_70:.4f}")
