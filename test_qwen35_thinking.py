"""Test Qwen3.5-0.8B thinking mode behavior."""
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3.5-0.8B"
CACHE_DIR = "/root/autodl-tmp/model"

try:
    from modelscope import snapshot_download
    model_path = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
    print(f"[OK] ModelScope: {model_path}")
except Exception as e:
    print(f"[WARN] ModelScope failed ({e}), falling back to HF")
    model_path = MODEL_ID

from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

print("Loading model...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("Model loaded.\n")

img = Image.open("/root/autodl-tmp/data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg").convert("RGB")
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
    'Locate and segment every instance that belongs to the following categories '
    '"person", report bbox coordinates and masks in JSON format.<|im_end|>\n'
    "<|im_start|>assistant\n"
)

inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
prompt_len = inputs['input_ids'].shape[1]

# Test 1: Default (should be non-thinking)
print("=" * 60)
print("TEST 1: Default mode (no extra params)")
print("=" * 60)
gen_out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
text = processor.tokenizer.decode(gen_out[0, prompt_len:].tolist(), skip_special_tokens=False)
print(repr(text[:300]))
print(f"\n<think> tags found: {('</think>' in text or '<think>' in text)}")
print(f"```json found: {'```json' in text}")
print(f"bbox_2d found: {'bbox_2d' in text}")

# Test 2: enable_thinking=False (explicit)
print("\n" + "=" * 60)
print("TEST 2: enable_thinking=False")
print("=" * 60)
try:
    gen_out = model.generate(**inputs, max_new_tokens=256, do_sample=False, enable_thinking=False)
    text = processor.tokenizer.decode(gen_out[0, prompt_len:].tolist(), skip_special_tokens=False)
    print(repr(text[:300]))
    print(f"\n<think> tags found: {('</think>' in text or '<think>' in text)}")
    print(f"```json found: {'```json' in text}")
    print(f"bbox_2d found: {'bbox_2d' in text}")
except TypeError as e:
    print(f"Kwon't support enable_thinking: {e}")

# Test 3: enable_thinking=True (explicit thinking)
print("\n" + "=" * 60)
print("TEST 3: enable_thinking=True")
print("=" * 60)
try:
    gen_out = model.generate(**inputs, max_new_tokens=256, do_sample=False, enable_thinking=True)
    text = processor.tokenizer.decode(gen_out[0, prompt_len:].tolist(), skip_special_tokens=False)
    print(repr(text[:300]))
    print(f"\n<think> tags found: {('</think>' in text or '<think>' in text)}")
    print(f"```json found: {'```json' in text}")
except TypeError as e:
    print(f"Kwon't support enable_thinking: {e}")

print("\n" + "=" * 60)
print("DONE")
