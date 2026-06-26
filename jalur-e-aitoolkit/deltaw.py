import sys, torch
from safetensors import safe_open
def delta_weight(path, alpha=128, rank=128):
    scale = alpha/rank
    mods = {}
    with safe_open(path,"pt") as h:
        for k in h.keys():
            if k.endswith("lora_A.weight") or k.endswith("lora_B.weight"):
                base = k.rsplit(".lora_",1)[0]
                mods.setdefault(base,{})[k.split(".lora_")[1][0]] = h.get_tensor(k).float()
    total = 0.0
    for base,d in mods.items():
        if "A" in d and "B" in d:
            A=d["A"]; B=d["B"]          # A:[r,in] B:[out,r]
            # ||scale * B@A||_F^2 = scale^2 * trace((B^T B)(A A^T))
            BtB = B.t() @ B             # [r,r]
            AAt = A @ A.t()             # [r,r]
            fro2 = (BtB * AAt.t()).sum().item() * (scale**2)
            total += fro2**0.5
    return total, len(mods)
for p,label in [
    ("/cache/models/winner-5FW2Eaae/checkpoints/last.safetensors","WINNER(pemenang)"),
    ("/ephemeral/output/ema_off/ema_off_000000036.safetensors","EMA-off c36"),
    ("/ephemeral/output/ema_off/ema_off_000000072.safetensors","EMA-off c72"),
    ("/ephemeral/output/ema_off/ema_off.safetensors","EMA-off c108"),
]:
    dw,n = delta_weight(p)
    print(f"{label:20s} Δweight={dw:9.2f}  ({n} modul)")
