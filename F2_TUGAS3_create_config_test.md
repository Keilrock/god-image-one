# F2 TUGAS 3 (revisi) — Test via ENTRYPOINT ala-validator (CPU, no-GPU, no-train)

> Revisi: validator GAK panggil create_config langsung. Dia build Dockerfile → ENTRYPOINT →
> CLI args → create_config → train. Jadi test lewat jalur NYATA itu (drive `image_trainer.main()`
> dengan CLI args), bukan fungsi terisolasi.

## 1. Gimana ENTRYPOINT parse args → create_config (hasil baca kode)

`run_image_trainer.sh` → `python3 image_trainer.py "$@"` → `asyncio.run(main())`.
`main()` (image_trainer.py L575+):
1. `argparse`: `--task-id --model --dataset-zip --model-type{sdxl,flux,qwen-image,z-image} --expected-repo-name --trigger-word --hours-to-complete`.
2. `model_path = get_image_base_model_path(--model)`
3. `prepare_dataset(...)` — unzip dataset (dari `CACHE_DATASETS_DIR/{task_id}_tourn.zip`) ke `{IMAGE_CONTAINER_IMAGES_PATH}/{task_id}/img/{repeat}_lora style/` (repeat=1 Qwen/Z, 5 SDXL).
4. `get_image_training_config_template_path()` → pilih template + is_style.
5. `auto_caption_dataset(train_data_dir, cat_str)` — **LLaVA, SELALU jalan** (append ke caption).
6. `config_path = create_config(task_id, model_path, --model, --model-type, --expected-repo-name, --trigger-word)`
7. `run_training(--model-type, config_path)` — train loop.

## 2. Dry-run mungkin? → TIDAK ada flag bawaan
`create_config` dan `run_training` **berurutan** di `main()`, **tanpa flag stop**. create_config TIDAK
terpisah dari train via CLI. **Cara: intercept** — mock `run_training` (stop sebelum train loop) +
mock `auto_caption_dataset` (LLaVA butuh model 13GB+GPU). Selain itu `main()` jalan ASLI:
argparse → prepare_dataset (unzip beneran) → template select → **create_config ASLI** → [stop].
Harness: `/ephemeral/f2test/entrypoint_sim.py` (sandbox path di-monkeypatch ke /ephemeral, stub fiber/transformers).

## 3. Hasil — drive `main()` via CLI args, stop setelah create_config

| skenario (CLI) | --model-type | --trigger-word | →detect | **use_ema** | **steps** | **cd** |
|---|---|---|---|---|---|---|
| Qwen **PERSON** (9img) | qwen-image | "David Miller" | person | **false** | **108** (12×9) | **ABSENT** |
| Qwen ART (no trigger) | qwen-image | *(none)* | art | true | 2600 (size-aware) | 0.05 |
| Qwen LOGO | qwen-image | "Brandmark_Essentials" | logo | true | 3000 (size-aware) | 0.05 |
| Qwen SOCIAL | qwen-image | "AuraVerse" | social | true | 2200 (size-aware) | 0.05 |
| Z-IMAGE (data person!) | z-image | "David Miller" | — (guard) | ABSENT (template Z) | 2000 (size-aware) | 0.05 |
| SDXL (branch toml) | sdxl | "MyTrigger" | default | n/a (toml) | epochs (toml) | 0.05 |

Hasil entrypoint **IDENTIK** dengan test create_config terisolasi sebelumnya → wiring CLI→create_config benar.

## 4. Penegasan: CUMA Qwen-person kena recipe khusus, sisanya UTUH
- ✅ **Qwen PERSON** → EMA-off + steps `12×img` (108) + no caption_dropout. (satu-satunya)
- ✅ **Qwen ART** (no trigger) → EMA-**on** + size-aware (2600) + cd0.05. **else branch, utuh.**
- ✅ **Qwen LOGO / SOCIAL** (punya trigger) → detect logo/social → **else branch** (EMA-on, step lama), **BUKAN** recipe person. Buktiin guard `category=='person'` bener.
- ✅ **Z-IMAGE** (sengaja dikasih data person + trigger) → **tidak kena** (guard `model_type==qwen-image`). size-aware 2000, EMA template.
- ✅ **SDXL** → branch toml, `qwen_person` gak pernah disentuh. category=default→plain-64, cd0.05. No crash.

## 5. 🔴 CAVEAT JUJUR (penting)
- **auto_caption (LLaVA) di-MOCK** (no-op) — butuh model 13GB + GPU. Jadi detektor di test ini baca
  **caption MENTAH**. Di produksi, LLaVA **append** teks ke caption SEBELUM create_config → caption final
  = mentah + LLaVA. Risiko: fraksi keyword bisa geser. **Mitigasi:** keyword person/logo/social ada di
  caption asli dgn fraksi tinggi (78–100%), LLaVA cuma NAMBAH teks (buat person malah nambah kata manusia)
  → kecil kemungkinan turun <60%. **TAPI belum diuji pada caption LLaVA-augmented** → wajib re-cek pas
  re-validate GPU (saat LLaVA beneran jalan).
- Re-validate GPU (1 run/task) tetap perlu utuk konfirmasi skor akhir.

## 6. Catatan teknis
- Harness CPU (entrypoint_sim.py + stub) di `/ephemeral/f2test/` (ephemeral, gak di-commit). Logika detektor dijaga unit test `scripts/core/tests/test_category_detector.py` (committed).
- Z template (`base_diffusion_zimage.yaml`) emang tak punya `ema_config` → `use_ema` ABSENT = kondisi asli template.
- Sweep inert: dijalankan tanpa `DETHRONE_SWEEP_*` → `_apply_sweep_overrides_aitoolkit` no-op → F2 yang berlaku.
