# LACAK ASAL-USUL recipe SDXL-logo size-bucket (CPU, baca git/file doang)

> Tujuan: cari dari MANA angka recipe SDXL-logo size-bucket (ep/min_snr/batch/d_coef) datang —
> hasil training/sweep terukur, tebakan, atau warisan juara? Branch jalur-e-aitoolkit. NO GPU, NO train.
> Patokan: GIT_HISTORY_MAP.md, VERIFIKASI_flux_qwenart_sdxllogo.md.

---

## 1. `person_config.json` — struktur & riwayat
- **33 entri model**, 28 signature distinct → **bucket per-model** (beda tiap base SDXL).
- Riwayat **cuma 2 commit**: `c65c405` (import boss 05-29) → `6db821a` (Wen 06-04). **TIDAK disentuh jalur-c/d/e.**
- Buckets: xs (1-10 img), s (11-20), m (21-30), l (31-50), xl (51-1000). Tiap bucket: max_train_epochs,
  train_batch_size, unet_lr/te_lr, lr_scheduler, min_snr_gamma, prior_loss_weight, optimizer prodigy (d_coef), dll.

## 2. Apa yang Wen ubah (commit 6db821a, 06-04)?
**CUMA NAMBAH blok `default` fallback** (8 insertion, 2 deletion): `"default": {}` → `default:{xs..xl}`
(dipakai HANYA kalau model gak ada di `data`). **Blok `data` per-model TIDAK diubah sama sekali.**
Tidak ada rationale/hasil-training di commit message ("Update person_config.json").

## 3. 🔴 VONIS ASAL ANGKA BUCKET = (c) WARISAN BOSS

Banding `data[model].s` boss (c65c405) vs HEAD — **IDENTIK di semua model**:

| model | boss 's' bucket | HEAD 's' bucket | identik? |
|---|---|---|---|
| sdxl-base | ep60 / min_snr6 / batch6 / d_coef1.0 / prior0.78 | sama | ✅ |
| protovision-xl | ep32 / min_snr6 / batch6 / d_coef1.2 | sama | ✅ |
| anima-pencil | ep45 / min_snr6 / batch8 / d_coef1.1 | sama | ✅ |
| realvis-v4 | ep27 / min_snr5 / batch6 / d_coef1.2 | sama | ✅ |

→ Angka bucket = **boss verbatim** dari `c65c405` ("Tournament winner repository - Commit: 97607210").
**BUKAN** sweep kita, **BUKAN** tebakan, **BUKAN** Wen. Boss yang tuning & validasi lewat kemenangan turnamen mereka.
Sweep jalur-c kita (`DETHRONE_SWEEP_*`) **cuma** variasiin network(DoRA/plain)+cd lewat env — **gak pernah** nyentuh
bucket (riwayat file 2-commit buktiin). **Bukti:** boss==HEAD identik + history cuma c65c405→6db821a(default doang).

## 4. 🔴 KOREKSI — "ep60" itu ARTEFAK TEST, bukan recipe nyata
Di VERIFIKASI sebelumnya aku pakai `--model stabilityai/stable-diffusion-xl-base-1.0` (placeholder asal) →
sdxl-base 's' = **ep60**. Itu **BUKAN** task logo nyata. Task logo asli (dari dethrone_tasks.json + notes):
- **T5** (menang −16.9%): base = **protovision-xl-v6.6**, trigger "Brandmark Blueprint", boss 0.04106 → 's' = **ep32**.
- **#6** (dataset 7489f54c, 18 img, dipake reproduksi jalur-e): base = **anima-pencil** → 's' = **ep45**.

→ Epoch gantung (model, img-bucket); semua nilai = warisan boss. **ep60 gak pernah dipakai task logo manapun.**
**Jawaban "ep60 == winning T5?": TIDAK** — T5 pakai protovision **ep32**.

## 5. BANDING vs BOSS (c65c405 = boss verbatim, ADA di repo kita)
Boss `c65c405` image_trainer.py: **0** match `SDXL_NETWORK_BY_CATEGORY|detect_image_category|dora_wd|lycoris`
→ boss **GAK punya routing, plain LoRA**. `caption_dropout_rate = 0.1` (L307, semua toml branch).

| field | KITA (jalur-c) | BOSS (c65c405) | status |
|---|---|---|---|
| network | DoRA32 + conv4/4 + dora_wd + dropout0 | **plain `networks.lora` 32/32 no-conv** | 🔴 BEDA (deviasi jalur-c, via routing a5ddde9) |
| caption_dropout | **0.05** | **0.1** | 🔴 BEDA (jalur-c) |
| epoch (bucket) | warisan boss (anima45 / proto32) | sama | ✅ IDENTIK |
| min_snr / batch / d_coef / lr / optimizer / scheduler | warisan boss | sama | ✅ IDENTIK |

Notes jalur-c sendiri konfirmasi (`RECIPE_sdxl_logo.md:45`): *"LR/optimizer/scheduler/steps/min_snr KITA == BOSS PERSIS… satu-satunya beda = NETWORK"* (+ cd).

→ **VONIS recipe SDXL-logo kita = (b) BOSS + tweak network(DoRA)+cd(0.05) kita.** Epoch/optimizer/bucket = murni warisan boss.

## 6. 🔴 TAPI — WIN BELUM TER-REPRODUKSI (bukti hasil training kita sendiri)
Walau epoch = warisan boss, kemenangannya **belum aman**:
- `RESULTS_sdxl_logo.md`: reproduksi #6 (anima, bucket 's' ep45) → **GAGAL**. best ep25 = **0.0557 > boss 0.0465**;
  full ep45 makin overfit (text loss 0.074→0.103).
- Diagnosa: **boss menang lewat WINDOW-CUT** — boss "last" = checkpoint AWAL (~**epoch 10** = 0.04556), bukan ep45 penuh
  (SDXL-logo window 0.5-1.0h motong training). Kita train penuh → overfit.
- Jadi bucket-epoch (45/32) itu **MAX**; angka efektif yang menang = **early checkpoint hasil window-cut**, bukan epoch penuh.

---

## 📋 RINGKAS VONIS
1. **Asal angka bucket:** ✅ **(c) WARISAN BOSS** — boss verbatim (c65c405); Wen (6db821a) cuma nambah `default` fallback.
   Boss-validated, bukan sweep/tebakan kita. **Bukti:** boss==HEAD identik semua model + riwayat file 2 commit.
2. **ep60:** artefak placeholder test (sdxl-base). Task logo nyata: T5=protovision **ep32**, #6=anima **ep45**.
3. **Recipe kita vs boss:** epoch/optimizer/bucket = boss warisan (identik); **beda cuma network (DoRA+conv) + cd (0.05 vs 0.1)** — deviasi sengaja jalur-c.
4. 🔴 **Win belum aman:** recipe "kebawa" (DoRA+cd0.05+bucket-boss), TAPI kemenangan boss bergantung
   **window-cut early-checkpoint (~ep10)** yang **jalur-e GAGAL reproduksi** (overfit). SDXL-logo = satu-satunya task yang **masih kalah / open**.

### Sumber/bukti
- `scripts/lrs/person_config.json` (HEAD), `git show c65c405:` & `6db821a` (boss vs Wen).
- `git show c65c405:scripts/image_trainer.py` (boss no-routing, cd 0.1 L307).
- `dethrone_tasks.json` (T5 base protovision), `jalur-e-aitoolkit/RESULTS_sdxl_logo.md` & `RECIPE_sdxl_logo.md` (#6 anima, reproduksi gagal, window-cut diagnosa).
