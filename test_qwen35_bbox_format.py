"""Test Qwen3.5-0.8B bbox grounding output format."""
import sys
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3.5-0.8B"
CACHE_DIR = "/root/autodl-tmp/model"

# ModelScope download
try:
    from modelscope import snapshot_download
    model_path = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
    print(f"[OK] ModelScope: {model_path}")
except Exception as e:
    print(f"[WARN] ModelScope failed ({e}), falling back to HF")
    model_path = MODEL_ID

# Load
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

print("Loading model...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("Model loaded.\n")

# Load a VOC person image
img_path = "/root/autodl-tmp/data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg"
img = Image.open(img_path).convert("RGB")
print(f"Image: {img_path} ({img.size})\n")

# Build grounding prompt (same format as training)
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
    'Locate and segment every instance that belongs to the following categories '
    '"person", report bbox coordinates and masks in JSON format.<|im_end|>\n'
    "<|im_start|>assistant\n"
)

inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)

# Generate
print("=" * 60)
print("GENERATING (default mode, non-thinking)...")
print("=" * 60)

gen_out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
prompt_len = inputs['input_ids'].shape[1]
generated_ids = gen_out[0, prompt_len:].tolist()
raw_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

print("\n--- RAW OUTPUT ---")
print(repr(raw_text))
print("\n--- CLEAN OUTPUT ---")
print(raw_text)

# Parse bbox
import re, json
json_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
if json_match:
    print("\n--- FOUND JSON BLOCK ---")
    print(json_match.group(1))
    try:
        data = json.loads(json_match.group(1))
        for i, item in enumerate(data):
            bbox = item.get("bbox_2d")
            print(f"  Instance {i}: bbox={bbox}, label={item.get('label')}")
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}")
else:
    print("\n--- NO ```json block found, trying regex ---")
    matches = re.findall(r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', raw_text)
    if matches:
        for i, m in enumerate(matches):
            print(f"  Instance {i}: bbox=[{m[0]}, {m[1]}, {m[2]}, {m[3]}]")
    else:
        print("  No bbox_2d found in output!")
