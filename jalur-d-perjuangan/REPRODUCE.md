# Reproduce — OneTrainer + DoRA for Qwen/Z (jalur-d gate)

Exact setup + command sequence used for GATE 1. **The config + this sequence is what matters for reproduce, not the artifact.**

## 0. Environment notes
- L40 gate rig: driver 535 / CUDA 12.2 → could NOT use OneTrainer's pinned `torch 2.12.0+cu130` (needs driver ≥580).
- **H100 (Fase 2): driver ≥580 → use OneTrainer's pinned `torch 2.12.0+cu130` and SKIP the L40 torch downgrade + the adamw shim below.**
- Heavy data on `/ephemeral` (713GB scratch); root is only 97GB. Validator image is 48.9GB.

## 1. OneTrainer + isolated venv
```bash
git clone --depth 1 https://github.com/Nerogar/OneTrainer.git        # HEAD used: 07254ad
pip install --user virtualenv && python3 -m virtualenv /root/ot-venv  # (no apt python3-venv needed)

# --- L40 (CUDA 12.x) variant: ---
/root/ot-venv/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# --- H100 variant (preferred, matches OneTrainer pin): ---
# /root/ot-venv/bin/pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 --index-url https://download.pytorch.org/whl/cu130

/root/ot-venv/bin/pip install -r OneTrainer/requirements-global.txt           # transformers 5.5.4, diffusers-git, accelerate, optimizers
/root/ot-venv/bin/pip install bitsandbytes==0.49.1 onnxruntime-gpu==1.23.2    # int8/fp8 + onnx (skip cu130 torch/nccl pins)
/root/ot-venv/bin/pip install hf_transfer einops kornia spandrel              # fast dl + comfy loadtest deps
mkdir -p OneTrainer/training_samples && echo "[]" > OneTrainer/training_samples/samples.json
```

### L40-ONLY patch (torch 2.11 API drift; NOT needed on torch 2.12)
In `modules/util/optimizer/adamw_extensions.py` AND `adam_extensions.py`, replace
`self._cuda_graph_capture_health_check()` with:
```python
(getattr(self, "_cuda_graph_capture_health_check", None) or getattr(self, "_accelerator_graph_capture_health_check", lambda: None))()
```

## 2. Base models (diffusers format)
```bash
HF_HOME=/ephemeral/hf /root/ot-venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen-Image', local_dir='/ephemeral/models/Qwen-Image', ignore_patterns=['*.md','*.png','*.jpg'])"   # ~54GB
HF_HOME=/ephemeral/hf /root/ot-venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Tongyi-MAI/Z-Image', local_dir='/ephemeral/models/Z-Image', ignore_patterns=['*.md','*.png','*.jpg'])"    # ~20GB
```

## 3. Train DoRA (saves safetensors directly — no separate convert step)
```bash
cd OneTrainer
HF_HOME=/ephemeral/hf /root/ot-venv/bin/python scripts/train.py --config-path onetrainer-configs/config_qwen_dora.json
HF_HOME=/ephemeral/hf /root/ot-venv/bin/python scripts/train.py --config-path onetrainer-configs/config_z_dora.json
# outputs: /ephemeral/work/out/dora_qwen.safetensors , dora_z.safetensors
```
(`output_model_format: SAFETENSORS` → OneTrainer writes the ComfyUI-loadable safetensors directly; no conversion needed.)

## 4. ComfyUI key-map load-test (in validator container)
```bash
docker pull gradientsio/image-evaluator:basilica
docker run --rm --gpus all -e COMFY_DIR=/app/validator/evaluation/ComfyUI \
  -v /ephemeral/work:/work -v /ephemeral/work/out:/loras -v /ephemeral/models/Qwen-Image:/qwen \
  --entrypoint python gradientsio/image-evaluator:basilica \
  /work/loadtest.py --lora /loras/dora_qwen.safetensors --transformer-dir /qwen/transformer
# PASS = patches == DoRA modules AND 'lora key not loaded' == 0   (Qwen 720/720, Z 210/210)
```
`loadtest.py` is committed in `onetrainer-configs/`.
