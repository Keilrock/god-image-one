# F3 FASE 0 — AUDIT titik integrasi OneTrainer (Z-Image)

> Tujuan: SEBELUM nulis kode integrasi OneTrainer, petakan entrypoint + routing + Dockerfile existing
> biar integrasi nempel di titik bener, gak ngerusak Qwen (ai-toolkit). NO GPU/train, baca doang.
> Branch jalur-e-aitoolkit. Admin (diagonalge) approve backend bebas; output WAJIB safetensors ComfyUI-loadable
> + train selesai dalam window. Target: Z-Image → OneTrainer DoRA (recipe menang 0.0308). Qwen TETAP ai-toolkit.

---

## A. ENTRYPOINT & titik percabangan backend
- `dockerfiles/standalone-image-toolkit-trainer.dockerfile`: `FROM diagonalge/ai-toolkit:latest` →
  `ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]` → `image_trainer.py main()`. **OneTrainer GAK ada di image ini.**
- `main()` (L607-650): parse CLI → `model_path` → `prepare_dataset` → **`auto_caption` LLaVA (L639)** → `create_config` → `run_training`.
- 🔴 **Backend percabangan ada di DUA titik, flag sama** `is_ai_toolkit = model_type in [Z_IMAGE, QWEN_IMAGE]`:
  1. **`create_config` L273** → build config: ai-toolkit **YAML** (z+qwen) vs kohya **TOML** (sdxl/flux).
  2. **`run_training` L543-550** → invoke: `python3 /app/ai-toolkit/run.py {config}` (z+qwen) vs `accelerate launch .../{type}_train_network.py` (kohya).
- → **Z & Qwen SEKARANG berbagi jalur ai-toolkit di KEDUA titik.** Integrasi = pisahin Z di 2 tempat itu.

## B. Z di-handle sekarang (ai-toolkit)
- `create_config` ai-toolkit branch (L275-318): load **`base_diffusion_zimage.yaml`** (plain LoRA linear32/alpha32, lr1e-4,
  adamw8bit, cd0.05, flowmatch — **BUKAN DoRA**), set `name_or_path`/`folder_path`/`trigger`, **size-aware steps** (Z cap 2000), save YAML.
- Invoke: `python3 /app/ai-toolkit/run.py {task_id}.yaml`.
- Output: `training_folder` = `get_checkpoints_output_path` = **`/app/checkpoints/{task_id}/{expected_repo_name}`** (sesuai spec validator).

## C. Aset OneTrainer jalur-d (`jalur-d-perjuangan/onetrainer-configs/`)
| file | isi | reusable? |
|---|---|---|
| **`Z_DoRA-1150_WINNER_0.0308_DO-NOT-CHANGE.json`** | **config MENANG**: epochs64, res**1024**, **eager (compile=False)**, stochastic_rounding, rank16/alpha16 **DoRA** (lora_decompose), lr3e-4 ADAMW, transformer FP8, LOGIT_NORMAL | ✅ **template utama** |
| `config_z_dora.json` | config reproduce — **BUKAN winner** (epochs**4**, res**512**, **compile=True**) | ⚠️ jangan mentah |
| `concepts.json` | definisi dataset OneTrainer (path concept, balancing REPEATS) | ✅ template |
| `loadtest.py` | verifier ComfyUI key-map (pakai comfy.lora validator asli) | ✅ verifikasi output |
| `onetrainer_deps.md` + `REPRODUCE.md` | resep install (venv isolasi) + command train | ✅ resep Dockerfile/build |

🔴 **OneTrainer framework SENDIRI gak ada di repo/Dockerfile** — mesti clone `Nerogar/OneTrainer @07254ad` + **venv terisolasi** `/root/ot-venv`.
🔴 **Semua path config = HARDCODE `/ephemeral/...`** (base_model `/ephemeral/models/Z-Image`, output `/ephemeral/work/out/dora_z.safetensors`,
  concepts `/ephemeral/work/...`) — manual jalur-d. **Wajib di-parameterize** ke path validator.
- Discrepancy winner vs reproduce-config: **epochs 64 vs 4, res 1024 vs 512, compile False vs True** → pakai **WINNER** sbg template (compile=True pernah crash → winner eager).

### Invoke OneTrainer (REPRODUCE.md §3)
```
/root/ot-venv/bin/python scripts/train.py --config-path <config.json>
# output_model_format SAFETENSORS -> tulis safetensors ComfyUI-loadable LANGSUNG (no convert)
```

## D. Output format & ComfyUI-loadable
- `output_model_format: SAFETENSORS` + `lora_decompose: true` (DoRA) → safetensors ComfyUI-loadable langsung, **no convert**.
- Key format: `transformer.layers.N.attention.to_k.{lora_down,lora_up,alpha,dora_scale}`.
- **GATE1 1A PASS** (container validator `gradientsio/image-evaluator:basilica`): Z comfy class **Lumina2**, **210/210 modul** ke-map,
  **0** `lora key not loaded`, `dora_scale` dihormati (`weight_decompose`). Verifikasi via `loadtest.py` (ComfyUI validator asli).
- 🔴 **GATE 1B (step-fit dalam window 0.5-1.0h) = TBD, belum pernah diukur di H100.** Feasibility muat cukup step BELUM dikonfirmasi.

## E. 🔴 REKOMENDASI titik integrasi (JANGAN diimplement — review Wen)
1. **Dockerfile** (`standalone-image-toolkit-trainer.dockerfile`): tambah blok — `git clone Nerogar/OneTrainer @07254ad`,
   bikin **`/root/ot-venv`** (virtualenv), install torch + `requirements-global.txt` + bitsandbytes/onnx **DI DALAM venv itu**.
   **JANGAN** system-pip. Pertahankan ai-toolkit + LLaVA.
2. **Routing `image_trainer.py`** — pisahin Z di 2 titik:
   - `create_config`: `elif model_type == z-image:` → bangun JSON OneTrainer dari template WINNER, inject
     `base_model_name=model_path`, `concept path=train_data_dir`, `output_model_destination=/app/checkpoints/{task_id}/{repo}`.
   - `run_training`: cabang z-image → `/root/ot-venv/bin/python /app/OneTrainer/scripts/train.py --config-path {config}`.
   - **Qwen tetap ai-toolkit** (ubah `is_ai_toolkit` jadi qwen-only / cabang z eksplisit duluan).
3. **Config**: fungsi baru `create_config_onetrainer()` (atau cabang) + template = WINNER json (parameterized path).
4. **Reuse**: WINNER config, concepts.json, loadtest.py, resep REPRODUCE. **Baru**: logika inject-path, langkah venv Dockerfile, cabang routing.

## 🔴 RISIKO ke Qwen (ai-toolkit) + isolasi
| risiko | dampak | mitigasi |
|---|---|---|
| Salah split `is_ai_toolkit` | Qwen ke-route ke OneTrainer → senjata utama rusak | cabang Z eksplisit; Qwen→ai-toolkit PERSIS |
| Dep konflik (OneTrainer torch2.12/transformers5.5.4 vs ai-toolkit) | install system-pip → ai-toolkit pecah → **Qwen mati** | **WAJIB venv `/root/ot-venv` terisolasi** (jalur-d buktiin jalan) |
| Build gagal di image SHARED | OneTrainer install error → image toolkit gagal → **Qwen + Z mati dua-duanya** | build defensif, pin versi, test build (risiko TERTINGGI) |
| torch cu130 vs cu128 | driver H100 validator belum diketahui (≥580?) → OneTrainer crash runtime | konfirmasi driver validator (UNRESOLVED) |
| LLaVA (L639) jalan buat Z | caption Z ke-modif LLaVA, beda dari run jalur-d (no LLaVA) | mungkin skip auto_caption buat z-OneTrainer (Fase 1) |
| compile=True crash | config reproduce compile=True pernah crash | pakai WINNER (eager) |

## VONIS Fase 0
Integrasi LAYAK (admin approve backend bebas; output ComfyUI-load PASS di GATE1 1A) dan titik nempel JELAS:
**isolasi via venv `/root/ot-venv` + cabang Z di create_config & run_training, Qwen ai-toolkit gak disentuh.**
2 hal **belum kelar** sebelum implement: (1) **driver H100 validator** (cu128 vs cu130), (2) **GATE 1B step-fit window** (belum diukur).
Risiko terbesar = build gagal di image bersama (Qwen ikut mati) → butuh build defensif + test build.

### Sumber/bukti
- `dockerfiles/standalone-image-toolkit-trainer.dockerfile`, `scripts/image_trainer.py` (L273 create_config split, L543-550 run_training split, L607-650 main).
- `jalur-d-perjuangan/onetrainer-configs/` (WINNER json, config_z_dora, concepts, loadtest.py, onetrainer_deps.md), `jalur-d-perjuangan/REPRODUCE.md`, `GATE1_RESULT.md`.
