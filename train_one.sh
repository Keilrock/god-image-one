#!/bin/bash
# train_one.sh — wrapper 1 training run buat sweep (ADAPTED utk VM L40 ini). 1 GPU, NO DDP.
# Dipanggil sweep_dethrone.py dgn env DETHRONE_SWEEP_* udah ke-set + arg posisi:
#   $1=model_type $2=base_model $3=train_data_dir $4=trigger $5=out_dir $6=gpu
# Output: LoRA safetensors di $out_dir (struktur yg di-push HF).
#
# === CARA KERJA (mirror orkestrasi produksi: trainer-downloader -> standalone-image-trainer) ===
# Trainer (scripts/image_trainer.py) TIDAK download apa-apa; dia baca dari path lokal:
#   - dataset zip : /cache/datasets/<task_id>_tourn.zip   (prepare_dataset unzip -> /dataset/images/...)
#   - base model  : /cache/models/<repo--dgn-dash>/        (get_image_base_model_path)
#   - adapter/clip: /cache/hf_cache/                       (Qwen/Z only)
#   - output LoRA : /app/checkpoints/<task_id>/<repo_name>/
# Jadi train_one.sh: (1) zip data lokal ke path itu, (2) pre-download base model (+adapter Qwen/Z)
# pakai fungsi trainer_downloader DI DALAM image yg sama (parity), (3) jalanin trainer, env sweep diteruskan.
#
# ENV opsional:
#   DETHRONE_IMAGE  (default standalone-image-trainer)  — image hasil rebuild branch jalur-c
#   DETHRONE_CACHE  (default $HOME/dethrone_cache)       — persist model/adapter antar-arm (reuse, gak re-download)
#   DETHRONE_HOURS  (default 2)                          — hours-to-complete (mimic 1xH100 budget tournament)
#   HF_TOKEN                                             — utk pull base model (gak di-paste ke chat)
set -euo pipefail

MODEL_TYPE="$1"; BASE="$2"; TRAIN_DIR="$3"; TRIGGER="${4:-}"; OUT="$5"; GPU="$6"

IMAGE="${DETHRONE_IMAGE:-standalone-image-trainer}"
CACHE="${DETHRONE_CACHE:-$HOME/dethrone_cache}"
HOURS="${DETHRONE_HOURS:-2}"
TASK_ID="$(basename "$OUT")"     # = nama arm (mis a_plain64_cd05) -> zip & extraction terisolasi per-arm
HERE="$(cd "$(dirname "$0")" && pwd)"   # repo dir (utk mount script patch tanpa rebuild image)

echo "[train_one] type=$MODEL_TYPE base=$BASE trigger='$TRIGGER' gpu=$GPU task_id=$TASK_ID"
echo "[train_one] DETHRONE env: $(env | grep DETHRONE_SWEEP_ || echo none)"
echo "[train_one] image=$IMAGE cache=$CACHE out=$OUT"

if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A "$TRAIN_DIR" 2>/dev/null)" ]; then
  echo "[train_one][FATAL] train_data_dir kosong/gak ada: $TRAIN_DIR" >&2; exit 2
fi

mkdir -p "$CACHE/datasets" "$CACHE/models" "$CACHE/hf_cache" "$OUT"

# --- 1) build dataset zip (flat: gambar + .txt caption) ke path yg dibaca image_trainer ---
# nama "..._tourn.zip" => prepare_dataset TIDAK hapus zip setelah extract (lihat prepare_diffusion_dataset.py).
ZIP="$CACHE/datasets/${TASK_ID}_tourn.zip"
rm -f "$ZIP"
# pakai python zipfile (host gak selalu punya `zip`); flat: semua file non-hidden di TRAIN_DIR.
python3 - "$TRAIN_DIR" "$ZIP" <<'PY'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
n = 0
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, src))
            n += 1
print(f"zipped {n} files -> {dst}")
PY
echo "[train_one] dataset zip: $ZIP ($(unzip -l "$ZIP" | tail -1))"

# --- 2) pre-download base model (+ adapter/clip/t5 utk Qwen/Z) ke cache, pakai fungsi produksi ---
# dijalanin DI DALAM image trainer biar versi huggingface_hub/transformers identik. skip kalau udah ada.
echo "[train_one] ensuring base model + deps in cache (skip kalau sudah ada)..."
docker run --rm -i --gpus "device=$GPU" --runtime nvidia \
  -e HF_TOKEN="${HF_TOKEN:-}" -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -v "$CACHE/models:/cache/models" -v "$CACHE/hf_cache:/cache/hf_cache" \
  --entrypoint python3 "$IMAGE" - "$BASE" "$MODEL_TYPE" <<'PY'
import sys, asyncio
sys.path.insert(0, "/workspace")
from core.models.utility_models import ImageModelType
import trainer.constants as cst
import trainer.utils.trainer_downloader as d

repo, mt = sys.argv[1], sys.argv[2]

async def go():
    p = await d.download_base_model(repo, cst.CACHE_MODELS_DIR, ImageModelType(mt))
    print(f"[downloader] base model -> {p}", flush=True)
    if mt == ImageModelType.Z_IMAGE.value:
        await d.download_adapter("ostris/zimage_turbo_training_adapter",
                                 "zimage_turbo_training_adapter_v2.safetensors", cst.HUGGINGFACE_CACHE_PATH)
    elif mt == ImageModelType.QWEN_IMAGE.value:
        await d.download_adapter("ostris/accuracy_recovery_adapters",
                                 "qwen_image_torchao_uint3.safetensors", cst.HUGGINGFACE_CACHE_PATH)
    if mt in (ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value):
        from transformers import CLIPTokenizer
        from huggingface_hub import snapshot_download
        CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
        CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
        snapshot_download(repo_id="google/t5-v1_1-xxl", repo_type="model",
                          cache_dir=cst.HUGGINGFACE_CACHE_PATH, local_dir_use_symlinks=False,
                          allow_patterns=["tokenizer_config.json", "spiece.model",
                                          "special_tokens_map.json", "config.json"])

asyncio.run(go())
PY

# --- 3) jalanin trainer (env sweep diteruskan; LoRA keluar langsung ke $OUT) ---
# expected-repo-name=output => output_dir = /app/checkpoints/<task_id>/output, di-mount ke $OUT.
echo "[train_one] launching trainer..."
docker run --rm --gpus "device=$GPU" --runtime nvidia \
  -e DETHRONE_SWEEP_NETWORK -e DETHRONE_SWEEP_CD -e DETHRONE_SWEEP_STEPS \
  -e DETHRONE_SWEEP_QWEN_TE -e DETHRONE_SWEEP_LINEAR \
  -e HF_TOKEN="${HF_TOKEN:-}" -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -e TRANSFORMERS_ALLOW_TORCH_LOAD=true \
  -v "$CACHE/models:/cache/models" \
  -v "$CACHE/datasets:/cache/datasets" \
  -v "$CACHE/hf_cache:/cache/hf_cache" \
  -v "$OUT:/app/checkpoints/${TASK_ID}/output" \
  -v "$HERE/scripts/image_trainer.py:/workspace/scripts/image_trainer.py:ro" \
  "$IMAGE" \
    --task-id "$TASK_ID" \
    --model "$BASE" \
    --dataset-zip "local" \
    --model-type "$MODEL_TYPE" \
    --trigger-word "$TRIGGER" \
    --expected-repo-name "output" \
    --hours-to-complete "$HOURS"

# --- 4) prune BACKEND-AGNOSTIK: ambil LoRA final -> checkpoints/last.safetensors ---
# kohya(SDXL): $OUT/last-000NNN.safetensors di root. ai-toolkit(Qwen/Z): subfolder, last_000NNNNNNN.safetensors.
# eval (eval_diffusion.find_latest_lora_submission_name) cuma liat file startswith "checkpoint" +
#   (endswith "last.safetensors" ATAU trailing-step tertinggi). Jadi taro 1 final di checkpoints/last.safetensors.
# rekursif + filter ketat ke pola 'last(-/_NNN)?.safetensors' (buang optimizer/ema/sample .safetensors).
mapfile -t STS < <(find "$OUT" -type f -name 'last*.safetensors' | grep -E '/last([_-][0-9]+)?\.safetensors$' | sort -V)
if [ "${#STS[@]}" -eq 0 ]; then
  echo "[train_one][FATAL] gak ada LoRA last*.safetensors di $OUT setelah training" >&2
  echo "[train_one] semua .safetensors di $OUT:" >&2; find "$OUT" -name '*.safetensors' >&2; exit 3
fi
FINAL=""
for f in "${STS[@]}"; do [ "$(basename "$f")" = "last.safetensors" ] && FINAL="$f"; done   # kohya final unnumbered
[ -z "$FINAL" ] && FINAL="${STS[-1]}"                                                        # else: step tertinggi
echo "[train_one] LoRA final dipilih: $FINAL (dari ${#STS[@]} kandidat: ${STS[*]##*/})"
TMP="$(mktemp -d)"; cp -f "$FINAL" "$TMP/last.safetensors"
rm -rf "${OUT:?}/"* 2>/dev/null || true
mkdir -p "$OUT/checkpoints"; mv -f "$TMP/last.safetensors" "$OUT/checkpoints/last.safetensors"; rmdir "$TMP" 2>/dev/null || true
echo "[train_one] DONE: pruned -> $(ls -la "$OUT"/checkpoints/last.safetensors)"
