#!/bin/bash
# train_one.sh — wrapper 1 training run buat sweep (ADAPT ke setup L40-mu). 1 GPU, NO DDP.
# Dipanggil sweep_dethrone.py dgn env DETHRONE_SWEEP_* udah ke-set + arg posisi:
#   $1=model_type $2=base_model $3=train_data_dir $4=trigger $5=out_dir $6=gpu
# Output: LoRA safetensors di $out_dir (struktur yg di-push HF).
set -e
MODEL_TYPE="$1"; BASE="$2"; TRAIN_DIR="$3"; TRIGGER="$4"; OUT="$5"; GPU="$6"
echo "[train_one] type=$MODEL_TYPE base=$BASE trigger='$TRIGGER' gpu=$GPU"
echo "[train_one] DETHRONE env: $(env | grep DETHRONE_ || echo none)"

# === TODO ADAPT: jalanin trainer container/-mu, teruskan env DETHRONE_SWEEP_* ===
# Contoh pola (sesuaikan volume + cara naro train_data + cara container baca dataset):
#   docker run --rm --gpus "device=$GPU" \
#     -e DETHRONE_SWEEP_NETWORK -e DETHRONE_SWEEP_CD -e DETHRONE_SWEEP_STEPS \
#     -e DETHRONE_SWEEP_QWEN_TE -e DETHRONE_SWEEP_LINEAR \
#     -v "$TRAIN_DIR:/dataset/images/<task>/img/5_lora style:ro" \
#     -v "$OUT:/app/checkpoints:rw" \
#     standalone-image-trainer \
#       --task-id sweep --model "$BASE" --dataset-zip "<local-or-url>" \
#       --model-type "$MODEL_TYPE" --trigger-word "$TRIGGER" --hours-to-complete 2
#
# WAJIB: container mesti pakai image hasil rebuild branch jalur-c (ada lycoris + env hooks).
echo "[train_one] >>> ISI perintah trainer L40-mu di sini <<<" >&2
exit 1
