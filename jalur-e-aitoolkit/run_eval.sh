#!/usr/bin/env bash
# Rekonstruksi eval runner jalur-e — container image-evaluator:basilica-redisfix.
# Replikasi validator: Qwen LoRA img2img di test set lokal, 10 seed (master 42), weighted 0.25*text+0.75*no_text.
#
# Usage:
#   run_eval.sh <MODELS_repo[,repo2]> [TEST_DIR] [LOCAL_LORA_SAFETENSORS:SAVE_NAME]
# Contoh Fase 2 (winner dari HF):
#   run_eval.sh gradients-io-tournaments/tournament-...-5FW2Eaae
# Contoh Fase 3 (LoRA lokal hasil train):
#   run_eval.sh fake/ema_off_c108 /ephemeral/datasets/qwen_t2_test/test_data /ephemeral/output/ema_off/ema_off_000000108.safetensors:ema_off_c108
set -euo pipefail

MODELS="$1"
TEST_DIR="${2:-/ephemeral/datasets/qwen_t2_test/test_data}"
LOCAL_LORA="${3:-}"
IMAGE="image-evaluator:basilica-redisfix"
BASE_REPO="gradients-io-tournaments/Qwen-Image"
MODEL_TYPE="qwen-image"

LORAS_HOST=/ephemeral/comfy/loras
DIFFM_HOST=/ephemeral/comfy/diffusion_models
mkdir -p "$LORAS_HOST" "$DIFFM_HOST"

# pre-place local LoRA so download_lora() short-circuits (cek file exist di LORAS_SAVE_PATH)
if [ -n "$LOCAL_LORA" ]; then
  SRC="${LOCAL_LORA%%:*}"; SAVE="${LOCAL_LORA##*:}"
  cp -f "$SRC" "$LORAS_HOST/$SAVE.safetensors"
  echo "[local-lora] $SRC -> $LORAS_HOST/$SAVE.safetensors"
fi

NAME="eval-$$-${RANDOM}"
echo "[run] image=$IMAGE models=$MODELS test=$TEST_DIR"
docker run --name "$NAME" --gpus all --runtime nvidia \
  -v "$(readlink -f "$TEST_DIR")":/workspace/input_data:ro \
  -v "$LORAS_HOST":/app/validator/evaluation/ComfyUI/models/loras \
  -v "$DIFFM_HOST":/app/validator/evaluation/ComfyUI/models/diffusion_models \
  -e DATASET=/workspace/input_data \
  -e ORIGINAL_MODEL_REPO="$BASE_REPO" \
  -e MODELS="$MODELS" \
  -e MODEL_TYPE="$MODEL_TYPE" \
  -e TRANSFORMERS_ALLOW_TORCH_LOAD=true \
  "$IMAGE"
RC=$?

OUT=/ephemeral/output/eval_$(echo "$MODELS" | tr '/,' '__')_$$.json
docker cp "$NAME:/aplp/evaluation_results.json" "$OUT" 2>/dev/null && echo "[result] $OUT" || echo "[result] FAILED to cp (rc=$RC)"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# print weighted
python3 - "$OUT" <<'PY' 2>/dev/null || true
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("no result json:",e); sys.exit()
for repo,r in d.items():
    if repo=="model_params_count":
        print("params:",r); continue
    if isinstance(r,dict) and "eval_loss" in r:
        el=r["eval_loss"]; tg=el.get("text_guided_losses") or []; nt=el.get("no_text_losses") or []
        ta=sum(tg)/len(tg) if tg else float("nan"); na=sum(nt)/len(nt) if nt else float("nan")
        w=0.25*ta+0.75*na
        print(f"{repo}\n  text={ta:.5f} no_text={na:.5f}  WEIGHTED={w:.5f}")
    else:
        print(repo, "->", r)
PY
echo "EVAL_DONE rc=$RC"
