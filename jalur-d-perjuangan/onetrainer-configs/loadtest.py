#!/usr/bin/env python
"""Faithful key-mapping load-test using the VALIDATOR'S OWN ComfyUI code.
Loads the Qwen base via comfy, builds the lora key_map exactly as the validator's
LoraLoader does (comfy.lora.model_lora_keys_unet), then applies the OneTrainer DoRA
via comfy.lora.load_lora @ strength 1.0 and reports matched vs 'lora key not loaded'.
"""
import sys, os, glob, argparse, logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
COMFY=os.environ.get("COMFY_DIR","/ephemeral/validator-src/validator/evaluation/ComfyUI")
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch
import comfy.utils, comfy.lora, comfy.sd

# capture "lora key not loaded" warnings emitted by comfy.lora.load_lora
NOT_LOADED=[]
class _Cap(logging.Handler):
    def emit(self, r):
        m=r.getMessage()
        if "lora key not loaded" in m: NOT_LOADED.append(m)
logging.getLogger().addHandler(_Cap())

ap=argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--transformer-dir", required=True, help="diffusers transformer dir of base")
args=ap.parse_args()

# 1) load OneTrainer DoRA
lora = comfy.utils.load_torch_file(args.lora, safe_load=True)
print(f"\n=== DoRA file: {args.lora} ===")
print("total tensors:", len(lora))
dora = [k for k in lora if k.endswith("dora_scale")]
print("dora_scale keys:", len(dora))
sample = list(lora.keys())[:6]
print("sample lora keys:", sample)

# 2) load Qwen base transformer via comfy (CPU) -> ModelPatcher, exactly like validator
files = sorted(glob.glob(os.path.join(args.transformer_dir, "*.safetensors")))
print("\n=== base transformer shards:", len(files))
sd = {}
for f in files:
    sd.update(comfy.utils.load_torch_file(f, safe_load=True))
print("base transformer tensors:", len(sd))

model = comfy.sd.load_diffusion_model_state_dict(sd)
if model is None:
    print("!!! comfy could NOT detect/load Qwen transformer -> cannot build keymap"); sys.exit(2)
print("comfy model class:", type(model.model).__name__)

# 3) build key_map exactly as validator's LoraLoader
key_map = {}
key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
print("model_lora_keys_unet entries:", len(key_map))

# 4) apply DoRA @ strength 1.0 (load_lora returns patch_dict; warns 'lora key not loaded')
loaded = comfy.lora.load_lora(lora, key_map, log_missing=True)
print("\n=== RESULT ===")
print("patches created (matched module groups):", len(loaded))
# recompute leftover precisely
matched_lora_keys=set()
# load_lora consumed keys internally; recompute by re-running matching
# count lora keys that map to a model key present in key_map
import comfy.weight_adapter as wa
mapped=0
for x in key_map:
    # any lora key starting with x.* present?
    if any(k==x or k.startswith(x+".") or k.startswith(x.replace(".weight",".")) for k in lora):
        mapped+=1
print(f"key_map entries with a matching lora key: ~{mapped}")
n_modules = sum(1 for k in lora if k.endswith("lora_up.weight"))
print("\n=== VERDICT ===")
print(f"DoRA modules in file (lora_up count): {n_modules}")
print(f"patches created: {len(loaded)}")
print(f"'lora key not loaded' warnings: {len(NOT_LOADED)}")
if NOT_LOADED[:10]:
    print("  examples:", NOT_LOADED[:10])
ok = len(loaded) >= n_modules and len(NOT_LOADED) == 0
print("KEY-MAP:", "PASS ✅" if ok else "FAIL ❌")
