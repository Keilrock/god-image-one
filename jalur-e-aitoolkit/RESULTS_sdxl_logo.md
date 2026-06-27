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
