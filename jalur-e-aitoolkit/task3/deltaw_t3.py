import torch
from safetensors import safe_open
def dw(path, alpha=128, rank=128):
    scale=alpha/rank; mods={}
    with safe_open(path,"pt") as h:
        for k in h.keys():
            if k.endswith("lora_A.weight") or k.endswith("lora_B.weight"):
                base=k.rsplit(".lora_",1)[0]; mods.setdefault(base,{})[k.split(".lora_")[1][0]]=h.get_tensor(k).float()
    tot=0.0
    for b,d in mods.items():
        if "A" in d and "B" in d:
            A=d["A"];B=d["B"]; tot+=((B.t()@B)*(A@A.t()).t()).sum().item()**0.5*scale
    return tot,len(mods)
for p,l in [
 ("/cache/models/winner-5GU4/checkpoints/last_000001000.safetensors","BOSS 5GU4 c1000 (EMA-on)"),
 ("/ephemeral/output_t3/ema_off_t3/ema_off_t3_000000600.safetensors","EMA-off c600"),
 ("/ephemeral/output_t3/ema_off_t3/ema_off_t3_000000800.safetensors","EMA-off c800"),
 ("/ephemeral/output_t3/ema_off_t3/ema_off_t3_000001000.safetensors","EMA-off c1000"),
 ("/ephemeral/output_t3/ema_off_t3/ema_off_t3.safetensors","EMA-off c1100"),
]:
    v,n=dw(p); print(f"{l:26s} Δweight={v:9.2f} ({n} modul)")
