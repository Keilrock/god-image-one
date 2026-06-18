# RUNBOOK DETHRONE — jalur-c-dethrone (L40)

Target: dethrone boss `5GU4Xkd3` di final round SN56 image. **Urutan: 0 → 2A → 2B → 1.**
**PRINSIP:** matching boss = TIE = boss menang (threshold 3%). Sweep berarti HANYA kalau nemu **edge >3% nyata** (`loss ≤ boss×0.97`). Gak nemu → **JANGAN submit, tahan**. 1 GPU/arm (mimic 1×H100, **JANGAN DDP**).

---

## PRA-SYARAT (sekali)
```bash
git checkout jalur-c-dethrone                      # HEAD = recipe dethrone
docker build -f dockerfiles/standalone-image-trainer.dockerfile -t standalone-image-trainer .
#   ^ cek log: "[dethrone] lycoris already present" ATAU install sukses.
#     kalau "[WARN] lycoris install GAGAL" -> DoRA fallback ke plain (style/logo/design nggak DoRA, tapi nggak crash)
export HF_TOKEN=...   ; export HF_NS=keilrockstars      # JANGAN paste ke chat
```
**Adapt `train_one.sh`** ke setup trainer L40-mu (teruskan env `DETHRONE_SWEEP_*`, pakai image hasil rebuild).

**Siapin data lokal per task** (presigned expire ~19 Jun — pull SEKARANG):
```bash
# tiap task: simpan JSON detail (gradients.io) -> task_<T>.json, lalu:
python3 fetch_task_dataset.py --json task_QF.json --out data/QF --split 0.2   # QF: 9 train / held-out
python3 fetch_task_dataset.py --json task_T1.json --out data/T1 --split 0.1   # T1 David 44/5
python3 fetch_task_dataset.py --json task_T5.json --out data/T5 --split 0.1   # T5 logo
python3 fetch_task_dataset.py --json task_T6.json --out data/T6 --split 0.1   # T6 (re-validasi logo)
# struktur dipakai: data/<T>/train , data/<T>/test
```

---

## ▶ FASE 0 — VALIDASI EVAL (no train, MURAH, WAJIB DULU)
Buktiin replikasi eval lokal akurat sebelum percaya sweep apapun. Set = **QF + T1 + T5** (2 backend + single/multi-img + gap tipis; T3 dibuang krn dobel SDXL sama T5).
```bash
python3 validate_eval.py --tasks QF T1 T5 --data-root data --gpu 0 --tol 0.004
```
Reproduce target: QF chal **0.0580** / me **0.0616** · T1 boss **0.08585** / chal **0.08671** · T5 boss **0.04106** / chal **0.04127**.

### 🛑 GATE
- **SEMUA OK (±0.004)** → lanjut FASE 2A.
- **MELESET jauh** → replikasi bug (base/model_type/EVAL_DEFAULTS/seed salah, atau test_data beda dr validator). **STOP, lapor.** Yang krusial: **RANKING** (boss < chal) harus konsisten walau absolut geser (held-out lokal ≠ held-out validator).

---

## ▶ FASE 2A — person-SDXL (de-risk cd/saturasi SEBELUM Qwen mahal)
QF data (blue-pencil Ramiro). Arms: `plain64 cd05`, `plain64 cd10`, `plain96`, `DoRA64`.
```bash
python3 sweep_dethrone.py --phase person-sdxl --task QF \
  --train-data-dir data/QF/train --test-dir data/QF/test --gpu 0
```
Baca **DELTA antar-arm di kolom no_text** (75% skor). Anchor: chal 0.0580 / Wen 0.0616. Cari arm yang nyata < itu.

---

## ▶ FASE 2B — person-Qwen (TARUHAN UTAMA: TE-on edge)
T1 data (David). Arms: `TEoff` (=boss), `TEon`, `TEon cd10`. **TE-on = lever yang boss & chal BELUM coba.**
```bash
python3 sweep_dethrone.py --phase person-qwen --task T1 \
  --train-data-dir data/T1/train --test-dir data/T1/test --gpu 0
```
Target: **≤ 0.0833** (boss 0.08585 × 0.97). Person = kategori paling sering (.25) → edge di sini paling bernilai.

---

## ▶ FASE 1 — logo-SDXL (sekunder/asuransi, .15 freq)
T5 data. Arms: `DoRA32`, `+LoRA+16`, `+LoRA+16 rank48`, `+cd0`.
```bash
python3 sweep_dethrone.py --phase logo --task T5 \
  --train-data-dir data/T5/train --test-dir data/T5/test --gpu 0
```
Target: **≤ 0.0398** (boss 0.04106 × 0.97). Arm menang → **re-validasi di T6** (atau gabung test T5+T6 = 6 img) biar bukan fit ke T5 doang:
```bash
# contoh re-validasi: train arm menang di T5/train, eval di test gabungan
cp data/T6/test/* data/T5_T6/test/ ; cp data/T5/test/* data/T5_T6/test/   # gabung 6 img
python3 sweep_dethrone.py --phase logo --task T5 --skip-train \
  --train-data-dir data/T5/train --test-dir data/T5_T6/test --gpu 0
```

---

## 🚦 GATE PRA-SUBMIT (tiap recipe yang mau dipakai)
1. **FASE 0 lulus** (replikasi valid).
2. Recipe punya **edge >3% nyata** di kategorinya (`≤ boss×0.97`), tervalidasi di ≥2 task / test gabungan. **Gak ada edge → TAHAN, jangan submit.**
3. Tiap LoRA hasil train: `python3 check_backend_keys.py <lora.safetensors> --model-type <sdxl|qwen-image|z-image>` → harus **OK** (anti LoRA-mati spt boss T6).
4. **Update commit-hash endpoint ke HEAD `jalur-c-dethrone`** sebelum submit.

## Catatan
- Default jalur-c = matching boss di Z-logo/Qwen-person/product (TIE→boss). Style = satu-satunya win bawaan.
- Test set kecil (1–5 img) → noise; baca DELTA antar-arm, bukan absolut; re-validasi task kedua.
- Kategori weights (terverifikasi constants): person .25 · style .25 · logo .15 · social .15 · design .10 · product .10.
