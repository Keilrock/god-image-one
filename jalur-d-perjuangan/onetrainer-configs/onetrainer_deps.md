# OneTrainer — Manual Deps & Build Steps (buat nyusun Dockerfile)

> Konteks: validator clone repo → build dari Dockerfile → train di container **TANPA internet**,
> **/cache read-only** (base model di-cache validator di `/cache/models/{model_id}`).
> Semua yang manual di VM ini WAJIB masuk Dockerfile, kalau enggak = gak keliatan validator → build/train gagal.
> Live doc — di-append tiap nemu dep baru. Last update: 2026-06-24.

## Environment VM ini (buat referensi)
- H100 80GB PCIE, driver **535.183.06 / CUDA 12.2** (BUKAN ≥580). Ubuntu 22.04, Python 3.10.12.
- ⚠️ Driver 535 = TIDAK bisa cu130 (CUDA 13). Maka pakai jalur cu128 + shim (lihat di bawah).

---

## 1. OS packages (apt)
| package | kenapa perlu |
|---|---|
| `python3.10-venv` | `python3 -m venv` gagal tanpa ini (ensurepip absent di Ubuntu base). Wajib buat bikin venv isolated. |
| `python3-pip` | pip buat venv (kalau base image belum ada). |

> Di Dockerfile pakai base image `python:3.10` / `nvidia/cuda:*-cudnn-runtime` → venv module biasanya udah ada,
> tapi `nvidia/cuda` base butuh `apt-get install -y python3-venv python3-pip`. Catat biar gak kelupaan.

## 2. PyTorch — OVERRIDE dari pin OneTrainer (KRITIS, driver-dependent)
OneTrainer `requirements-cuda.txt` nge-pin `torch==2.12.0+cu130` (butuh driver ≥580).
**VM ini driver 535 → cu130 mustahil.** Maka di-override ke jalur L40 yang proven:
```
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```
- Install `requirements-global.txt` SAJA (SKIP `requirements-cuda.txt` / `requirements.txt` yang nge-pull cu130).
- ⚠️ KEPUTUSAN DOCKERFILE: kalau validator H100 driver-nya **≥580** → pakai pin asli `torch 2.12.0+cu130`
  DAN **drop shim §3**. Kalau **<580** → pakai cu128 + shim. Branch ini harus dipastiin dari spek validator.

## 3. Code patch — shim optimizer (cuma kalau torch 2.11/cu128, §2)
torch 2.11 gak punya `Optimizer._cuda_graph_capture_health_check` (udah di-rename jadi
`_accelerator_graph_capture_health_check`). OneTrainer master masih manggil nama lama → crash pas step().
Patch 2 file (ganti pemanggilan, line 185 masing-masing):
```
sed -i 's/self\._cuda_graph_capture_health_check()/self._accelerator_graph_capture_health_check()/' \
  modules/util/optimizer/adam_extensions.py modules/util/optimizer/adamw_extensions.py
```
- Verified: torch 2.11.0+cu128 → `_cuda...`=False, `_accelerator...`=True. Patch wajib.
- Kalau jalur cu130 (torch 2.12) dipake → **JANGAN patch** (nama lama masih valid di sana).

## 4. pip (dari requirements-global.txt — otomatis, gak manual)
Ke-handle `pip install -r requirements-global.txt` (numpy, diffusers@git, transformers 5.5.4, accelerate,
optimizers, dll). Catatan: ada 3 editable git install (`diffusers`, `mgds`, `Muon`) → butuh `git` +
akses github PAS BUILD. Di container no-internet, ini harus udah ke-bake pas `docker build` (build-time
internet OK, run-time enggak). Pastiin build stage nge-resolve git deps ini.

## TODO buat Dockerfile (rangkuman)
- [ ] base image dgn CUDA cocok (cu128 atau cu130 tergantung driver validator)
- [ ] apt: python3-venv python3-pip git
- [ ] pip: torch override (§2) SEBELUM requirements-global
- [ ] pip: -r requirements-global.txt (resolve 3 git editable saat build)
- [ ] apply shim §3 (cuma jalur cu128)
- [ ] base model dari /cache (read-only) — JANGAN download di container

## 5. Required file — training_samples/samples.json (setup quirk)
OneTrainer crash di start (`to_pack_dict`) kalau `training_samples/samples.json` gak ada.
Wajib bikin file ini isi `[]` (relative ke OneTrainer root). Di Dockerfile:
`RUN mkdir -p training_samples && echo '[]' > training_samples/samples.json`
(Sumber: GATE1_RESULT.md config quirks — dikonfirmasi ulang di H100.)

## 6. bitsandbytes (buat ADAMW_8BIT / optimizer 8-bit)
OneTrainer requirements-global GAK include bitsandbytes. Optimizer `ADAMW_8BIT` (dipake recipe pemenang ai-toolkit) butuh `import bitsandbytes` -> crash `ModuleNotFoundError` kalau gak ada.
`pip install bitsandbytes` (di Dockerfile juga). Confirmed needed di H100 cu128.
