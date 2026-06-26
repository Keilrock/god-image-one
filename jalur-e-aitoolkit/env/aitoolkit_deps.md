# ai-toolkit deps — TERBUKTI JALAN (re-deploy 2026-06-26)

- ai-toolkit HEAD: `5f04ae7` (target 5f04ae7)
- venv: /ephemeral/venv-aitoolkit (virtualenv, python 3.10.12) — system python3-venv absent, pakai virtualenv
- GPU: H100 PCIe 80GB, driver 535.183.06 / CUDA 12.2 -> torch cu128 jalan, NO shim

## Key versions (pip)
```
accelerate                1.14.0
bitsandbytes              0.49.2
diffusers                 0.38.0.dev0
numpy                     1.26.4
optimum-quanto            0.2.4
peft                      0.18.1
safetensors               0.8.0
torch                     2.9.1+cu128
torchao                   0.10.0
torchaudio                2.9.1+cu128
torchvision               0.24.1+cu128
transformers              5.5.3
```

## Catatan re-deploy
- hf_xet di-uninstall dari venv (xet backend crash 'Background writer channel closed'); pakai HF https downloader.
- /cache & /root/.cache/huggingface di-symlink ke /ephemeral (root cuma 97GB, base 73GB gak muat).
- docker image 'image-evaluator:basilica-redisfix' (49GB) SURVIVE wipe (di root /var/lib/docker).
