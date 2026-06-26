# SESSION STATE — jalur-e (update 2026-06-26, sebelum matiin VM)

Posisi terkini biar besok gampang lanjut. Branch git: `jalur-e-aitoolkit`.

## Status per task

| Task | Kategori | Trainer | Status | Angka |
|------|----------|---------|--------|-------|
| **#2 Qwen person** | person, qwen-image | ai-toolkit | ✅ **MENANG** | EMA-off **0.06825** vs winner 0.07307 (−6.6%) |
| **#3 Qwen-art Jib-Mix** | art/style, qwen-image | ai-toolkit | ✅ **setara boss** (EMA-off kalah tipis) | EMA-off best 0.07344 vs boss 0.07292 → **pakai boss EMA-on** |
| **Z (logo)** | logo, z-image | OneTrainer | ✅ **MENANG** (jalur-d, JANGAN diutak) | DoRA-1150 **0.0308** vs boss 0.04083 (−24%) |
| **#4 Flux** | — | kohya | 🟡 **belum digarap — bahas besok** | — |
| **SDXL** | (product/logo/style) | kohya | ❓ belum digarap sesi ini | — |

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

## NEXT (besok)
1. **Flux #4** — bahas dulu sama Wen (recipe extract belum dilakuin). Reminder Wen: kohya (bukan ai-toolkit), rank 128/alpha 64, train_t5xxl true.
2. (opsional) SDXL tasks kalau masuk scope.
3. Re-setup env di VM baru (ikut pola `env/aitoolkit_deps.md` + symlink /cache).
