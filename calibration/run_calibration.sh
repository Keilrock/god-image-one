#!/usr/bin/env bash
# =============================================================================
# Epoch-calibration pilot — GPU-phase runner (ALL commands in one place).
# Run this ONLY on a GPU box (docker + nvidia runtime). CPU prep is already done.
#
#   bash calibration/run_calibration.sh
#
# What it does:
#   0. preflight checks
#   1. stage dataset (deterministic 11 train / 3 test)   [CPU, idempotent]
#   2. download nova-anime base model (.safetensors)
#   3. generate the calibration config (production recipe + epoch overrides)
#   4. TRAIN 40 epochs -> 10 LoRA checkpoints + sample grids
#   5. upload the 10 checkpoints to HF (1 repo each, public)
#   6. EVAL all 10 with the REAL validator scorer (img2img L2)
#   7. parse -> epoch->score table + knee
#
# Edit the CONFIG block below, then run. Stops on first error (set -e).
# =============================================================================
set -euo pipefail

# ---- CONFIG (edit these) ----------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_USER="${HF_USER:-keilrockstars}"           # HF account to host the 10 LoRA repos
GPU_ID="${GPU_ID:-0}"
TASK_ID="95f68a11-3b13-488f-8ed5-42fec8865160"
BASE_MODEL="John6666/nova-anime-xl-pony-v5-sdxl"

CALIB="$REPO/calibration"
DATA_ROOT="$CALIB/data"
TASK_DATA="$DATA_ROOT/$TASK_ID"
TRAIN_DATA_DIR="$TASK_DATA/img"                       # parent of "5_lora style"
TEST_DIR="$TASK_DATA/test"
OUT_DIR="$CALIB/checkpoints/${TASK_ID}_nova_s_calib"  # LoRA checkpoints land here
MODEL_DIR="$CALIB/base_model"
CONFIG="$CALIB/out/${TASK_ID}_nova_s_calib.toml"
RESULTS_DIR="$CALIB/eval_out"

# Docker images (override if needed)
TRAIN_IMAGE="${TRAIN_IMAGE:-diagonalge/kohya_latest:latest}"   # has accelerate + sd-scripts deps
VALI_IMAGE="${VALI_IMAGE:-gradientsio/image-evaluator:basilica}" # validator diffusion scorer
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
# -----------------------------------------------------------------------------

echo "==== PHASE 0: preflight ===="
command -v docker >/dev/null || { echo "docker missing"; exit 1; }
docker info 2>/dev/null | grep -qi nvidia || echo "WARN: nvidia runtime not detected in 'docker info' — GPU may be unavailable"
python3 -c "import huggingface_hub" || { echo "pip install huggingface_hub"; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "Run: huggingface-cli login"; exit 1; }
mkdir -p "$OUT_DIR" "$MODEL_DIR" "$RESULTS_DIR/aplp"

echo "==== PHASE 1: stage dataset (11 train / 3 test, deterministic) ===="
python3 "$CALIB/split_dataset.py" --out-root "$DATA_ROOT"
test "$(ls "$TRAIN_DATA_DIR/5_lora style/"*.png | wc -l)" -eq 11
test "$(ls "$TEST_DIR/"*.png | wc -l)" -eq 3

echo "==== PHASE 2: download base model ($BASE_MODEL) ===="
# nova-anime is published as a Diffusers FOLDER (model_index.json + unet/ vae/
# text_encoder*/ ...), NOT a single-file SDXL .safetensors. sd-scripts'
# sdxl_train_network loads a Diffusers directory natively: _load_target_model()
# branches on os.path.isfile(path) — a directory triggers
# StableDiffusionXLPipeline.from_pretrained() and converts the UNet to original
# SDXL internally. So we fetch the whole snapshot and point the trainer at the
# folder (an earlier single-file heuristic grabbed text_encoder/ only -> all
# UNet keys missing at load).
MODEL_PATH="$MODEL_DIR/diffusers"
python3 - "$BASE_MODEL" "$MODEL_PATH" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
p = snapshot_download(
    repo_id=repo,
    local_dir=dst,
    cache_dir=os.environ.get("HF_HUB_CACHE") or None,  # reuse existing blob cache
    allow_patterns=[
        "model_index.json", "scheduler/*", "unet/*", "vae/*",
        "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
    ],
)
print(p)
PY
test -f "$MODEL_PATH/model_index.json" || { echo "model_index.json missing in $MODEL_PATH"; exit 1; }
test -f "$MODEL_PATH/unet/diffusion_pytorch_model.safetensors" || { echo "UNet weights missing in $MODEL_PATH/unet"; exit 1; }
echo "base model (Diffusers folder): $MODEL_PATH"

echo "==== PHASE 3: generate calibration config ===="
python3 "$CALIB/generate_calib_config.py" \
  --model-path "$MODEL_PATH" \
  --train-data-dir "$TRAIN_DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --output-name "nova_s_calib" \
  --out "$CONFIG" --preview

echo "==== PHASE 4: TRAIN (40 epochs -> 10 checkpoints + samples) ===="
# Runs sd-scripts inside the kohya image. Host paths are mounted 1:1 so the
# absolute paths inside $CONFIG resolve unchanged.
docker run --rm --runtime nvidia --gpus "\"device=$GPU_ID\"" \
  -v "$REPO":"$REPO" -v "$HF_HUB_CACHE":"$HF_HUB_CACHE" \
  -w "$REPO/scripts/sd-script" \
  -e HF_HOME="$HOME/.cache/huggingface" \
  "$TRAIN_IMAGE" \
  accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision bf16 \
    --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 \
    sdxl_train_network.py --config_file "$CONFIG"

echo "checkpoints produced:"; ls -la "$OUT_DIR"/*.safetensors

echo "==== PHASE 5: upload 10 checkpoints to HF (public, 1 repo each) ===="
python3 "$CALIB/upload_checkpoints.py" --ckpt-dir "$OUT_DIR" --hf-user "$HF_USER"
MODELS="$(python3 -c "import json,sys; m=json.load(open('$CALIB/out/repos.json')); print(','.join(sorted(m, key=lambda r:m[r])))")"
echo "MODELS=$MODELS"

echo "==== PHASE 6: EVAL all checkpoints with the REAL validator scorer ===="
# Direct container run with a LOCAL test set (bypasses the expired S3 test_split_url).
# eval_diffusion reads DATASET first; results bind-mounted out via /aplp.
docker run --rm --runtime nvidia --gpus "\"device=$GPU_ID\"" \
  -v "$TEST_DIR":/workspace/input_data:ro \
  -v "$HF_HUB_CACHE":/app/validator/evaluation/ComfyUI/models/checkpoints \
  -v "$HF_HUB_CACHE":/app/validator/evaluation/ComfyUI/models/diffusers \
  -v "$RESULTS_DIR/aplp":/aplp \
  -e DATASET=/workspace/input_data \
  -e MODELS="$MODELS" \
  -e ORIGINAL_MODEL_REPO="$BASE_MODEL" \
  -e MODEL_TYPE=sdxl \
  -e TRANSFORMERS_ALLOW_TORCH_LOAD=true \
  "$VALI_IMAGE"

echo "==== PHASE 7: parse -> epoch -> score table ===="
python3 "$CALIB/parse_results.py" \
  --results "$RESULTS_DIR/aplp/evaluation_results.json" \
  --repos "$CALIB/out/repos.json"

echo ""
echo "DONE. Sample grids for visual check: $OUT_DIR/sample/  (per checkpoint epoch)"
echo "Raw scores: $RESULTS_DIR/aplp/evaluation_results.json"
echo "After picking the knee epoch, optionally retrain that epoch on ALL 14 images for the production recipe."
