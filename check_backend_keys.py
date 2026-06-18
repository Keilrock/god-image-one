#!/usr/bin/env python3
"""
GUARD anti-LoRA-mati (kegagalan boss T6: submit DiT-LoRA di task SDXL -> key mismatch -> loss 0.0649).
Cek nama tensor di .safetensors COCOK sama backend yg diharapkan SEBELUM submit. CPU-only, instan.

  python3 check_backend_keys.py <lora.safetensors> --model-type sdxl
  exit 0 = OK cocok ; exit 1 = MISMATCH (JANGAN submit)
"""
import argparse, json, struct, sys

EXPECT = {
    # kohya SDXL/Flux -> UNet keys
    "sdxl": {"good": ("lora_unet_", "lora_te1_", "lora_te2_"), "bad": ("diffusion_model.", "transformer.")},
    "flux": {"good": ("lora_unet_", "lora_te1_"), "bad": ()},
    # ai-toolkit DiT -> transformer keys
    "qwen-image": {"good": ("diffusion_model.", "transformer."), "bad": ("lora_unet_",)},
    "z-image": {"good": ("diffusion_model.", "transformer.", "lora_unet_"), "bad": ()},
}


def tensor_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    return [k for k in hdr if k != "__metadata__"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lora")
    ap.add_argument("--model-type", required=True, choices=list(EXPECT))
    args = ap.parse_args()

    keys = tensor_keys(args.lora)
    exp = EXPECT[args.model_type]
    n_good = sum(1 for k in keys if any(k.startswith(p) for p in exp["good"]))
    n_bad = sum(1 for k in keys if any(k.startswith(p) for p in exp["bad"]))
    sample = keys[0] if keys else "(kosong)"
    print(f"model_type={args.model_type}  tensors={len(keys)}  good_prefix={n_good}  bad_prefix={n_bad}")
    print(f"contoh key: {sample}")
    if n_good == 0 or n_bad > 0:
        print(f"!!! MISMATCH — LoRA ini TIDAK cocok utk {args.model_type}. JANGAN SUBMIT (bakal LoRA mati spt boss T6).")
        sys.exit(1)
    print(f"OK — key cocok backend {args.model_type}. Aman submit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
