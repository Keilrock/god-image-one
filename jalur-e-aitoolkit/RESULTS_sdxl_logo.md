# SDXL LOGO — FASE B RESULT

## FASE B1 — SETUP + SANITY ✅ (2026-06-27)

### Env
- **Env kohya SHARE sama Flux**: docker `diagonalge/kohya_latest` punya `sdxl_train_network.py` + **lycoris keinstall** (buat H1 DoRA). Gak perlu env baru.
- Eval: `image-evaluator:basilica-redisfix` (sama kayak flux), MODEL_TYPE=sdxl. Eval CEPET ~2.5s/prompt.

### Base + dataset
- Base `zenless-lab/sdxl-anima-pencil-xl-v5` = **diffusers format** (unet 5.14GB + te + vae), download 6.5GB → /ephemeral/sdxl_base/anima-pencil.
- Dataset logo #6: train `7489f54c` **18 img + 18 txt** (caption built-in) → kohya `5_lora style/` (SDXL repeats=5). test `eb52ad46` **3 img**.

### 🎯 SANITY (gerbang) — PASS
| model | text | no_text | WEIGHTED kita | OFFICIAL | delta |
|-------|------|---------|---------------|----------|-------|
| **BOSS logo #6** (5GU4) | 0.04804 | 0.04601 | **0.04652** | 0.04653 | **−0.0%** (persis) |
→ Pipeline SDXL/kohya VALID. 3 test img, single-seed master_seed 42, weighted 0.25*text+0.75*no_text.

### Recipe yg bakal di-train (LR/opt/step IDENTIK boss, beda CUMA network)
- Base template: `base_diffusion_sdxl_person.toml` (logo = person route, is_style False — confirmed boss mapping 228 person).
- lrs bucket "s" (18 img): **prodigy** (d_coef 1.1, decouple, wd 0.01, bias_correction, safeguard_warmup), unet_lr/te_lr 0.95, batch 8, grad_accum 1, **45 epoch**, lr_scheduler constant, **min_snr_gamma 6**.
- H1 (kita): lycoris.kohya DoRA 32/32 + conv4/conv_alpha4 + dora_wd, dropout 0, no loraplus.
- H2 (boss): networks.lora plain 32/32, NO conv, no loraplus.

### 📋 Lever config boss logo (referensi keputusan EDGE nanti — yg boss PAKAI):
- optimizer **prodigy** (d_coef 1.1), lr_scheduler **constant**, **min_snr_gamma 6**, **scale_weight_norms 5**.
- caption_dropout_rate **0**, network_dropout **0**, **noise_offset TIDAK ADA** (person template; style template ada noise_offset).
- clip_skip 1, max_token_length 75, resolution 1024, bf16, no_half_vae.
→ Lever boss BELUM pakai (kandidat edge kalau perlu): noise_offset, caption_dropout>0, min_snr beda, ema.

## STATUS B1: SELESAI — env OK, base+dataset ready, sanity PERSIS. STOP, tunggu konfirmasi buat B2 (train H1 vs H2).

---

## FASE B2 — HEAD-TO-HEAD H1 vs H2 ⚠️ REPRODUKSI GAGAL (2026-06-27)

### Throughput gate
- H1 (DoRA) gate 20 step: **~1.94 s/it**. Full 540 step ≈ ~17 min/run. SDXL cepet. (H2 plain ~1.74 s/it.)

### Hasil (test 3 img, master_seed 42, weighted 0.25*text+0.75*no_text)
| ckpt | H1 (DoRA+conv) | H2 (boss plain) |
|------|----------------|------------------|
| epoch 25 | **0.05569** (best H1) | **0.05567** (best H2) |
| epoch 30 | 0.05607 | 0.05782 |
| epoch 35 | 0.05834 | 0.06149 |
| epoch 40 | 0.05828 | 0.06041 |
| last (ep45) | 0.05927 | 0.06516 |
| **BOSS #6 (sanity)** | **0.04652** | (target) |

### 🔴 VERDICT: GAGAL — H2 GAK reproduce boss, H1≈H2 (network BUKAN lever)
1. **H2 (replikasi boss plain) GAGAL reproduce 0.04653** → best 0.05567 (epoch25), last 0.06516. Meleset +20-40%.
2. **H1 ≈ H2 di best (0.05569 vs 0.05567)** → DoRA+conv vs plain **GAK ada beda** signifikan. Network BUKAN lever yg bikin boss menang. (Fase-A hipotesis "DoRA mudharat" → ternyata network gak ngaruh; dua-duanya sama-sama jelek.)
3. **Dua-duanya OVERFIT**: best epoch 25, makin naik abis itu. **text loss meledak** (0.074→0.103) — lora ngancurin prompt-following. Boss text cuma 0.048.
4. Pipeline VALID (sanity boss 0.04652 persis) → angka kita bener, masalah di TRAINING kita.

### 🔬 Diagnosis (kenapa kita overfit, boss enggak?) — HIPOTESIS, belum diuji
Recipe yg kita rekonstruksi (person bucket "s": prodigy d_coef1.1, lr0.95, 45 epoch, batch8, repeats5)
overtrains. Boss "last" = 0.04653 (gak overfit). Kandidat akar masalah:
- **A. Boss "last" ke-cut window LEBIH AWAL** (SDXL logo window 0.5-1.0h) → boss "last" = checkpoint zona bagus (~epoch 10-20), BUKAN 45. Kita train penuh 45 → overfit. (boss HF ckpt: last-000005..040+last — bisa jadi last=epoch≤40 hasil cut.) Tapi: bahkan epoch25 kita (0.0557) > boss 0.0465, jadi cut-timing aja gak cukup jelasin.
- **B. Bucket/LR salah rekonstruksi**: mungkin #6 bukan bucket "s" (18 img), atau prodigy d_coef ketinggian → LR overshoot → overfit + text rusak. (data[anima] lrs kita==boss, TAPI size_key mungkin beda kalau img count efektif beda.)
- **C. Repeats/effective-steps mismatch**: 18×5/8 = ~11 step/epoch × 45 = ~495 step. Kalau boss efektif < itu → kita over-train.
- **D. prodigy konvergen beda** vs setup boss (seed/init) — tapi gap terlalu besar buat cuma variance.

### Lever boss BELUM pakai (referensi edge — TAPI diagnosa dulu, jangan tebak):
noise_offset (none), caption_dropout (0), ema (none), min_snr beda. NAMUN: masalah utama = OVERFIT + text rusak,
bukan kurang-regularisasi doang. Prioritas = cari kenapa text-following ancur (kemungkinan LR prodigy ketinggian / over-train), BUKAN tambah lever.

### STATUS B2: STOP — reproduksi gagal, network bukan lever. Butuh diagnosa akar (window-cut? LR? bucket?) sebelum lanjut. NUNGGU putusan Wen.
