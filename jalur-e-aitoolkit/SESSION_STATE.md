# SESSION STATE — jalur-e (update 2026-06-26, sebelum matiin VM)

Posisi terkini biar besok gampang lanjut. Branch git: `jalur-e-aitoolkit`.

## Status per task

| Task | Kategori | Trainer | Status | Angka |
|------|----------|---------|--------|-------|
| **#2 Qwen person** | person, qwen-image | ai-toolkit | ✅ **MENANG** | EMA-off **0.06825** vs winner 0.07307 (−6.6%) |
| **#3 Qwen-art Jib-Mix** | art/style, qwen-image | ai-toolkit | ✅ **setara boss** (EMA-off kalah tipis) | EMA-off best 0.07344 vs boss 0.07292 → **pakai boss EMA-on** |
| **Z (logo)** | logo, z-image | OneTrainer | ✅ **MENANG** (jalur-d, JANGAN diutak) | DoRA-1150 **0.0308** vs boss 0.04083 (−24%) |
| **#4 Flux social** | social/style, flux | kohya | ✅ **MENANG** | epoch120 **0.03507** vs juara 5FW2 0.0364 (−3.8%) vs boss 0.0372 (−5.6%) |
| **SDXL logo** | logo, sdxl | kohya | 🟡 **front terakhir — next session** | — |

## Temuan kunci sesi ini
- **EMA-off = lever PERSON-specific (low-step)**. Menang telak buat person #2 (108 step: EMA-on crush magnitude → EMA-off benerin). GAK transfer ke art #3 (1000 step: magnitude EMA-on udah sehat, EMA-off cuma over-shoot tanpa koherensi).
- **Boss-art #3 window-limited**: config tulis 2500 step tapi ke-cut di **1000** (last_000001000) oleh window 1.25j. Winner = step 1000. Submission step-scaling art: ~1000 cukup.
- **Magnitude ≠ loss** (echo OneTrainer): Δweight EMA-off > boss di art tapi loss tetep kalah.
- Pipeline eval VALID di 2 base (Qwen-Image + Qwen-Image-Jib-Mix), reproduce boss persis (+0.4%).

## Env / infra (di /ephemeral, ke-wipe pas VM mati — re-setup besok)
- venv-aitoolkit: torch 2.9.1+cu128, ai-toolkit 5f04ae7. Detail: `env/aitoolkit_deps.md`.
- Docker: `image-evaluator:basilica-redisfix` (49GB, SURVIVE wipe di root /var/lib/docker).
- Eval runner: `run_eval.sh` (MODEL_TYPE qwen-image, mount loras+diffusion_models ke ephemeral).
- ⚠️ /cache & HF cache HARUS symlink ke /ephemeral (root cuma 97GB, base 73GB gak muat).
- ⚠️ hf_xet uninstall dari venv (crash); pakai HF https downloader.

## Backup status
- **Git** (branch jalur-e-aitoolkit): semua config + RESULTS + eval json + deltaw + SESSION_STATE. CLEAN & pushed.
- **HF**:
  - `keilrockstars/qwen-emaoff-jalur-e` (private) — task #2: 3 checkpoint EMA-off + config + RESULTS ✅
  - `keilrockstars/qwen-art-jalur-e-task3` (private) — task #3: EMA-off c1000/c1100 + recipe boss EMA-on + RESULTS

## Flux #4 — BERES ✅ (2026-06-27)
- **Submission pick: `last-000120.safetensors` (0.03507)** — nyalip juara 5FW2 (0.0364, −3.8%) & boss (0.0372, −5.6%).
- Recipe = template boss VERBATIM (rank128/alpha64, Lion wd0.005, unet_lr 8e-5/te 8e-6, guidance 85, flow 3.1582, sigmoid, train_t5xxl, 1024, seed 2, 250 step). flux.json default udah di-revert ke `{}` (leftover yatim 4 Jun).
- Env: docker `diagonalge/kohya_latest` (sd-scripts BUILT-IN image, BUKAN vendored). Eval: `gradientsio/image-evaluator:basilica` + redisfix (pip install redis asyncpg minio boto3 ...).
- Sanity reproduce PERSIS: 5FW2 0.03636 (off 0.036437), boss 0.03717 (off 0.037169). Pipeline flux VALID.
- Menang walau recipe identik = non-determinism run-to-run (bf16+Lion+xformers). 1 run, variance band ~0.035-0.037, margin ke boss lebar.
- Detail lengkap: `RESULTS_flux_task4.md` + `RECIPE_flux_task4.md`.
- **HF backup**: `keilrockstars/flux-task4-jalur-e` (private) — ckpt 110/120/125 + config + RESULTS + eval json.

## NEXT (next session)
1. **SDXL logo** — front terakhir. (Recipe SDXL: lihat `scripts/lrs/style_config.json`/`person_config.json` + dethrone routing di image_trainer.py. Logo = lycoris DoRA jalur-d MENANG di z-image; SDXL logo recipe perlu di-cek.)
2. Re-setup env VM baru: docker kohya (flux) udah terbukti; ai-toolkit ikut `env/aitoolkit_deps.md` + symlink /cache.
