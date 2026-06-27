# AUDIT TODO — yang mesti dicek next session (CPU, NO GPU)

Dipicu temuan kritis 2026-06-27: eksperimen logo pakai config MANUAL, bukan pipeline asli → kelewat runtime
override (caption_dropout line 499/307, LLaVA auto-caption, network routing). Audit apakah task LAIN kena juga.

═══════════════════════════════════════════════
## 1. AUDIT per-task: MANUAL atau PIPELINE ASLI? Setting kelewat?
═══════════════════════════════════════════════
Buat tiap task, banding config yg DIPAKAI vs config yg pipeline asli (`image_trainer.py` → `create_config()`) generate.

| Task | Trainer | Cara train sesi ini | Risiko kelewat | Status audit |
|------|---------|---------------------|----------------|--------------|
| **Flux #4** | kohya | config_full.toml MANUAL | caption_dropout? LLaVA caption? | ❓ CEK |
| #2 Qwen person (EMA-off) | ai-toolkit | ? (sesi lama) | aitoolkit runtime override? | ❓ CEK |
| #3 Qwen-art | ai-toolkit | ? (sesi lama) | sda | ❓ CEK |
| SDXL logo | kohya | config MANUAL (cd=0) — SALAH, udah ke-detect | cd, llava, network | ✅ ke-detect, fix pending |

### Cara audit (no GPU): jalanin `create_config()` asli per task, banding vs config yg dipake.
Harness contoh (dipakai buat logo, lihat pipe_dump): mount repo + dataset, panggil `IT.create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word)`, dump toml. Banding field-by-field.

### 🔴 FLUX #4 — audit prioritas:
- Flux pakai `base_diffusion_flux.toml` + flux.json override. create_config flux: `for k,v in lrs_settings: config[k]=v`.
- 🔴 Cek: image_trainer.py ADA `config["caption_dropout_rate"]=...` buat flux? (line 307 boss = 0.1 buat SEMUA toml branch — termasuk flux?). Kalau iya, flux config_full.toml MANUAL gue (caption_dropout 0.1 ADA? cek) mungkin kelewat juga.
  - flux config_full.toml gue: `caption_dropout_rate = 0.1` (ADA, gue masukin dari template flux). Template flux gak punya cd, tapi gue set 0.1. CEK apakah cocok pipeline.
- 🔴 Cek: flux ada LLaVA auto-caption? main() panggil auto_caption_dataset buat SEMUA model_type non-aitoolkit. Flux train zip ADA caption built-in (11 txt). Apakah pipeline RE-CAPTION flux pakai llava? Kalau iya, training caption gue (built-in .txt) ≠ pipeline (llava-augmented).
  - TAPI: flux MENANG (0.03507) + sanity reproduce PERSIS → kemungkinan caption gak ngubah banyak ATAU llava di-skip. Konfirmasi.
- Verdict flux: kalau audit nunjukin manual ≈ pipeline (atau beda gak signifikan krn flux menang) → AMAN. Kalau beda → re-run flux via pipeline buat confirm 0.03507 valid.

### Qwen EMA-off (#2) / Qwen-art (#3) — ai-toolkit:
- ai-toolkit pakai yaml template + override beda (bukan toml/lrs). create_config aitoolkit branch (line ~278-340).
- Cek: ada runtime override (steps size-aware line 300, sweep env line 357) yg masuk/kelewat?
- Hasil #2/#3 dari sesi lama (EMA-off menang) — config-nya manual atau pipeline? Cek RESULTS_emaoff.md + eval_emaoff.

═══════════════════════════════════════════════
## 2. LOGO — fix arah mana? (keputusan, bukan audit)
═══════════════════════════════════════════════
Pipeline asli kita ≠ boss (2 deviasi dethrone):
- caption_dropout: kita **0.05** vs boss **0.1**
- network: kita **DoRA+conv** vs boss **plain LoRA**

Data sejauh ini: network irrelevant (H1≈H2), cd belum di-isolasi bener (manual cd=0 kacau).
**Hipotesis arah fix** (semua VIA PIPELINE ASLI, no manual):
- **Opsi A (samain boss persis)**: ubah pipeline → logo plain LoRA + cd0.1. Replikasi boss (proven 0.0465). Lewat env `DETHRONE_SWEEP_CD=0.1` + network override, atau edit image_trainer.py.
- **Opsi B (pipeline as-is)**: re-run DoRA+cd0.05+llava → liat landing. Kalau ~0.046 → cukup. Kalau kalah → ke A.
- **Opsi C (cari edge nyalip)**: setelah reproduce boss, sweep cd / best-early-checkpoint pick (boss submit epoch~10, bukan last!).
- 🔴 INGAT: boss submit best-EARLY checkpoint (epoch 10 = 0.0456), BUKAN last. Checkpoint-selection = bagian recipe. Pipeline kita simpan tiap 5 epoch → bisa pilih best.
- ⚠️ Butuh LLaVA 13GB buat pipeline asli (auto_caption). Build trainer image penuh.

═══════════════════════════════════════════════
## 3. LIST RUNTIME OVERRIDE PIPELINE ASLI per model_type (referensi)
═══════════════════════════════════════════════
Dari `scripts/image_trainer.py` create_config (jalur-e). Yang config MANUAL gampang kelewat:

### SDXL (toml branch):
- L462 pretrained_model_name_or_path, L463 train_data_dir, L467 output_dir
- L485-488 network (module/dim/alpha/args) ← dari SDXL_NETWORK_BY_CATEGORY[category], category auto-detect
- **L499 caption_dropout_rate = _cd_default (0.05; 0.10 person-default-route; env DETHRONE_SWEEP_CD override)**
- L340/357 lrs size-bucket apply (LR/optimizer/epochs/batch/min_snr)
- main(): prepare_dataset + **auto_caption_dataset (LLaVA, /opt/models/llava)** ← RE-CAPTION
- vs BOSS: L307 caption_dropout=0.1 (flat), network dari config_mapping (plain LoRA, no DoRA routing)

### Flux (toml branch):
- network dari template (gak ada category override flux). lrs flux.json override (default {} setelah revert).
- caption_dropout: CEK apakah L499/L307 kena flux juga. (AUDIT)
- auto_caption LLaVA: CEK apakah flux ke-recaption. (AUDIT)

### ai-toolkit (Qwen/Z, yaml branch L278-340):
- L300 size-aware steps (compute_aitoolkit_steps), L357 sweep env override (DETHRONE_SWEEP_*)
- caption_dropout di datasets (L261 sweep). CEK default.

═══════════════════════════════════════════════
## ATURAN AUDIT
═══════════════════════════════════════════════
- NO GPU buat audit (config-gen doang). GPU cuma kalau perlu re-train buat confirm.
- JANGAN commit config manual logo (cd=0 + dethrone-deviasi) sbg config aktif. NOTES only.
- Semua eksperimen ke depan VIA PIPELINE ASLI (`create_config()` / image_trainer.py), bukan toml manual.
