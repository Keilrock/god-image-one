# DETHRONE FULL RECAP — SN56 image, branch `jalur-c-dethrone` (L40 run)

Eksekusi sweep tervalidasi di VM L40 (1×L40 48GB, mimic 1×H100, no DDP). Tujuan: cari edge >3% vs boss
`5GU4Xkd3` per kategori. **Keputusan akhir: OPSI A (kompetitif / kerok emisi), BUKAN dethrone penuh.**
Tournament final: Kamis 15:00 UTC (endpoint di-update manual oleh operator).

---

## 0. Ringkasan keputusan

- **Style** = win bawaan (T4). **Logo** = edge nyata ketemu (FASE 1). Sisanya **fortress** (seri/concede).
- Dethrone penuh butuh **4/6 outright win**. Hasil sweep: 2 kategori kuat (style+logo), person-Qwen & product
  = fortress terkonfirmasi, person-SDXL tak terverifikasi vs tournament-boss. **4/6 tidak tercapai → ambil A.**
- Recipe jalur-c **sudah = recipe lock** (sweep MENGKONFIRMASI default jalur-c optimal di antara arm yang diuji;
  semua alternatif ditolak). Tidak ada perubahan recipe yang diperlukan.

---

## 1. Fase yang dikerjakan + tujuan

| Fase | Kategori/backend | Task | Tujuan |
|---|---|---|---|
| **0** | validasi eval (no-train) | QF, T1, T5 | Buktiin replikasi eval lokal akurat sebelum percaya sweep apapun |
| **2A** | person-SDXL | QF (blue-pencil) | De-risk lever cd/saturasi & rank/DoRA SEBELUM Qwen mahal |
| **2B** | person-Qwen | T1 (David) | Uji hipotesis TE-on (lever yg boss & chal "belum coba") |
| **1** | logo-SDXL | T5→T6 | Cari edge logo (flip gap 0.5% jadi >3%) + anti-fit re-validasi |

Urutan eksekusi: 0 → 2A → 2B → 1 (de-risk murah dulu, Qwen mahal di tengah, logo terakhir).

---

## 2. Hasil tiap fase (angka lengkap)

### FASE 0 — validasi eval (±0.004), LULUS TELAK (semua diff ≤0.00007)
| task | who | reproduce | expected | diff |
|---|---|---|---|---|
| QF (person/SDXL) | chal | 0.05801 | 0.05800 | +0.00001 |
| | me | 0.06158 | 0.06160 | −0.00002 |
| T1 (person/Qwen) | boss | 0.08578 | 0.08585 | −0.00007 |
| | chal | 0.08666 | 0.08671 | −0.00005 |
| T5 (logo/SDXL) | boss | 0.04107 | 0.04106 | +0.00001 |
| | chal | 0.04126 | 0.04127 | −0.00001 |

Ranking boss<chal konsisten di T1 & T5; gap QF chal<me ke-reproduce arahnya. **Eval lokal = mirror validator.**

### FASE 2A — person-SDXL (QF, eval 1 img #8). Skor = 0.25·text + 0.75·no_text.
| arm | text | no_text | weighted |
|---|---|---|---|
| a plain64 cd05 (=baseline/me) | 0.06219 | 0.05924 | 0.05998 |
| **b plain64 cd10** | 0.05535 | **0.05473** | **0.05489 (BEST)** |
| c plain96 cd05 | 0.05825 | 0.06067 | 0.06006 |
| d DoRA64 cd10 | 0.05592 | 0.05920 | 0.05838 |

anchor: chal 0.0580 / me 0.0616. Temuan: **cd05→cd10 nurunin no_text −0.00451** (lever nyata di SDXL).
rank 64→96 = nol/negatif. DoRA64 < plain64 (DoRA gak bantu person). *(QF = qualifier, bukan task tournament.)*

### FASE 2B — person-Qwen (T1, eval 5 img). Step LOCKED 2500 (terkonfirmasi dari log, bukan ke-potong patch-B).
| arm | text | no_text | weighted | vs boss 0.08585 |
|---|---|---|---|---|
| a TEon 2500 cd05 | 0.09997 | 0.08429 | 0.08821 | +0.00236 (lebih jelek) |
| b TEon 2500 cd10 | 0.11272 | 0.08815 | 0.09429 | +0.00844 (lebih jelek) |

target ≤0.08327. **Dua arm di ATAS boss → gak ada edge → TAHAN.** cd10 NAIKIN no_text (+0.0039) & ancur text
(+0.0128) → **lever cd KEBALIK di Qwen vs SDXL.** TE-on no-op (lihat §3). person-Qwen = **fortress**.

### FASE 1 — logo-SDXL. T5 (eval 3 img) + re-validasi T6 (anti-fit, brand beda).
| arm | T5 weighted | T6 weighted | vs boss T5 (0.04106) |
|---|---|---|---|
| a DoRA32 (baseline) | 0.03465 | 0.02781 | −15.6% ✅ |
| **b DoRA32+LoRA+16** | **0.03414** | 0.02666 | **−16.9% ✅ (best T5)** |
| c DoRA48+LoRA+16 | 0.03538 | 0.02769 | −13.8% ✅ |
| d DoRA32+cd0 | 0.03579 | **0.02582** | −12.8% ✅ (best T6, best text 2 task) |

target ≤0.03983. **Keempat arm nembus target di KEDUA task** (no train/test leakage, verified).
- **T5: menang bersih vs boss TERVERIFIKASI** (boss 0.04106, FASE 0 reproduce eksak).
- **Anti-fit LULUS:** recipe konsisten rendah di T6 (brand beda) → bukan fit-T5.
- **Caveat:** boss T6 = BLUNDER (0.0649, LoRA mati, gak repeatable) → margin vs-boss gak bisa di-konfirmasi
  independen di T6. vs challenger T6 asli (0.0226) kita sedikit lebih jelek (0.0258) → T6 task floor-nya
  rendah. Jadi: **1 logo-task (T5) edge >3% terverifikasi + recipe robust**, tapi belum 2 task vs-boss-valid.
- **LoRA+16 = no-op** (T6: baseline-a 0.02781 vs c-LoRA+ 0.02769; b vs a di T5 cuma −0.0005).

---

## 3. Temuan kunci

1. **cd (caption_dropout) = lever SDXL-only.** Nurunin no_text di person-SDXL (2A, −0.0045). **KEBALIK di Qwen**
   (2B, +0.0039 no_text, ancur text). → cd10 jangan dibawa ke Qwen. cd global 0.05 tetap default.
2. **TE-on (Qwen) = NO-OP.** ai-toolkit qwen_image: `create LoRA for Text Encoder: 0 modules` — network LoRA
   cuma cover DiT (transformer), export 0 TE keys. `train_text_encoder:true` gak hasilin adapter TE deployable.
   **Menjelaskan kenapa boss & chal gak pakai: emang gak ke-support stack ini.** (Riset TE-LoRA = deferred.)
3. **rank = dead-end (person).** 64→96 gak bantu (2A c lebih jelek).
4. **DoRA = cuma style/logo.** person-SDXL DoRA64 < plain64 (2A). Konsisten T3-final (DoRA product kalah −26%).
5. **LoRA+ratio16 = no-op (logo).** FASE 1 buktiin LoRA+ gak nambah apa-apa di logo → BUANG.
6. **person-Qwen = fortress.** recipe kita ≈ boss tapi gak nembus; TE-on buntu; cd nyakitin. Concede.
7. **logo edge NYATA vs boss** (T5 −16.9%) + robust (T6), tapi cuma 1 task boss-valid (T6 boss blunder).

---

## 4. Status final per kategori + recipe lock

| kategori | backend | status | recipe LOCK (= jalur-c default, tervalidasi) |
|---|---|---|---|
| **style** (.25) | SDXL/kohya | **WIN** | DoRA32 + conv4 + `loraplus_lr_ratio=16` (lycoris.kohya, dora_wd) |
| **logo** (.15) | SDXL/kohya | **EDGE** (T5 −16.9%, robust) | DoRA32 + conv4 (DoRA tanpa LoRA+) — cd 0.05 |
| design (.10) | SDXL/kohya | (tak muncul) | DoRA32 + conv4 |
| person (.25) | SDXL/kohya | tie (QF only) | plain-LoRA 64 (no DoRA, no conv) — **cd 0.10 via detektor person** (product tetap cd05) |
| product (.10) | SDXL/kohya | **fortress (seri)** | plain-LoRA 64 — cd 0.05 |
| person (.25) | Qwen/ai-toolkit | **fortress** (concede) | linear128, TEoff, cd05, step size-aware 2000–3000 (boss config 2500 → aktual ke-cap ~1250 by-budget) |
| logo (.15) | Z/ai-toolkit | fortress (seri) | ai-toolkit standar (linear32+conv16) |
| cd global | SDXL | — | **0.05** (default), kecuali override sweep (inert di produksi) |

Routing tetap konsisten: **style SDXL = caption-keyword champion** (`detect_styles_in_prompts`, c65c405) · **logo/design = keyword high-precision
(≥60% caption)** · **default (person/product/social/ambigu) = plain-64** (asimetri downside: plain-default cuma
suboptimal, DoRA-default = blunder product −26%).

### Keputusan cd final per-kategori:
- **person-SDXL cd10** (temuan 2A): **DI-APPLY via detektor person high-precision** (commit `ac84a2c`).
  `is_person_dataset`: cd10 HANYA kalau `person_frac≥0.60 & product_frac<0.20` di route "default" → person→cd10,
  **product (juga "default") TETAP cd05** (fortress aman). Network tetap plain-64; cuma cd beda. Bias-aman:
  false-negative (person→cd05) OK, false-positive (product→cd10) dilarang. Smoke-test: person→cd10, product→cd05, ambiguous→cd05.
- **logo cd0** (arm-d): TIDAK di-lock (tetap cd05). Data wash di weighted (cd05 menang T5 +0.001, cd0 menang T6 +0.002);
  cd0 menang text 2 task tapi verified-win T5 (−16.9%) pakai cd05. Pilih cd05 (proven). cd0 = future text-render exp.

---

## 5. Verifikasi repo aman turnamen (production path)

Referensi proven: `jalur-b-sizeaware` (posisi 4 minggu lalu). DIFF jalur-c vs jalur-b:
- **File BARU (10, semua sweep infra STANDALONE, TIDAK di-import production, TIDAK di-COPY ke image):**
  `sweep_dethrone.py, eval_local.py, eval_ab.py, validate_eval.py, check_backend_keys.py, train_one.sh,
  split_dataset.py, dethrone_tasks.json, README_DETHRONE.md, DETHRONE_ANALYSIS.md, PATCH_NOTES_dethrone.md`.
  (Dockerfile cuma `COPY scripts/ trainer/` → file root sweep gak masuk image.)
- **File MODIFIED (production, perubahan INTENSIONAL = recipe lock):**
  - `scripts/image_trainer.py`: routing DoRA per-kategori, cd 0.1→0.05, step floor (per_image 100→200,
    min 600→2000), + sweep-hook env (`DETHRONE_SWEEP_*`). **Semua hook env-gated → inert kalau env UNSET.**
  - `trainer/utils/training_paths.py`: style-routing SDXL = **caption-keyword champion** (`detect_styles_in_prompts`, c65c405).
    trigger-null sempat dicoba lalu **DIBALIKIN** ke champion (proven anti-nyasar; trigger-null cuma didukung 1 task
    observasi). Z/Qwen = tambahan operator `_detect_style_from_captions` (dipertahankan).
  - `base_diffusion_qwen_image.yaml` / `_zimage.yaml`: +`caption_dropout_rate: 0.05`.
  - `dockerfile` / `requirements.txt`: +lycoris_lora (defensive, build gak boleh gagal).

**Guard env-unset (kondisi turnamen, validator gak set `DETHRONE_SWEEP_*`):**
- `_apply_sweep_overrides_aitoolkit`: `if all(x is None ...): return` → no-op.
- `_sweep_network_override`: return None → pakai `SDXL_NETWORK_BY_CATEGORY` (recipe lock).
- `_sweep_env("DETHRONE_SWEEP_CD", default=0.05)` → 0.05.
- Entry `run_image_trainer.sh` **identik** jalur-b; `main()` argparse **tidak berubah**.
- **Tidak ada path sweep hardcoded** (/ephemeral, DETHRONE_CACHE) di production.
- `check_backend_keys.py` = tool manual standalone (BUKAN di flow validator) → tak bisa blok submission.

**VERDICT: ✅ REPO AMAN BUAT TURNAMEN.** Production path (env unset) = jalur-b + recipe lock; sweep infra
terisolasi (standalone, env-gated, tak masuk image). Yang berbeda dari jalur-b cuma recipe yang disengaja.

---

## 6. Catatan operasional

- **GPU cost (≈, 1×L40):** FASE 0 ~2.5j · 2A ~3.2j (4 arm SDXL) · 2B ~8.5j (2 arm Qwen, ~3.5j/arm + eval ~100m)
  · FASE 1 ~6j (4 arm T5 + 4 arm T6 + 2 eval). **Total ~20 jam GPU.** Qwen eval ~100m/run (20 inferensi/img-model).
- **Infra fix sepanjang run (di sweep harness, bukan production):**
  - Docker data-root direlokasi `/var/lib/docker → /ephemeral/docker` (root 97G kepenuhan pas pull eval image;
    ephemeral 713G). daemon.json data-root + nvidia runtime preserved.
  - `train_one.sh`: implement penuh (zip→download→train→prune), **backend-agnostik** (kohya `last-NNNN` root +
    ai-toolkit `last_NNNNNNNNN` subfolder → keduanya jadi `checkpoints/last.safetensors`).
  - `eval_local.py`: pass `HF_TOKEN` ke eval container (repo sweep private) + redact di log.
  - LoRA repo HARUS `checkpoints/last.safetensors` (eval `find_latest_lora_submission_name`: startswith
    "checkpoint" + endswith "last.safetensors"); root `lora.safetensors` GAGAL.
  - Qwen butuh image **toolkit** (`diagonalge/ai-toolkit`), bukan image kohya.
  - TE-on bentrok `cache_text_embeddings` (ai-toolkit raise) → patch matiin caching TE pas TE-on (env-gated).
- **Anomali:** boss T6 BLUNDER 0.0649 (DiT-LoRA di task SDXL → key mismatch → LoRA mati); real boss-logo ~0.041.
  Sweep auto-pakai 0.0649 sbg boss T6 → ABAIKAN, pakai ~0.041.
- **Boss T1 step (koreksi dari asumsi awal):** `checkpoints/config.yaml` boss = `steps: 2500`, TAPI checkpoint cuma
  sampai `last_000001250` (keep-4) → boss **niat 2500, aktual ke-cap ~1250** (validator `hours_to_complete` ngecap
  by-budget). LoRA yg menang (0.08585) = checkpoint 1250-step. → floor Qwen 2000 inert di turnamen (ke-cap duluan).
- **Belum diuji (next time):** person-SDXL vs boss tournament asli (QF cuma qualifier, gak ada boss); riset
  apakah ai-toolkit qwen bisa di-config LoRA text-encoder beneran (TE-on real); logo edge di task ke-2 yg
  boss-nya VALID (T6 blunder bikin re-validasi vs-boss gak konklusif).

---

## 7. Recipe table final (buat submission, OPSI A)

```
style  (SDXL) : lycoris.kohya  dim32 alpha32  conv_dim4 conv_alpha4  algo=lora dora_wd=True loraplus_lr_ratio=16  cd0.05
logo   (SDXL) : lycoris.kohya  dim32 alpha32  conv_dim4 conv_alpha4  algo=lora dora_wd=True                       cd0.05
design (SDXL) : lycoris.kohya  dim32 alpha32  conv_dim4 conv_alpha4  algo=lora dora_wd=True                       cd0.05
person (SDXL) : networks.lora  dim64 alpha64  (no conv, no DoRA)                                                  cd0.05
product(SDXL) : networks.lora  dim64 alpha64  (no conv, no DoRA)                                                  cd0.05
person (Qwen) : ai-toolkit linear128  TEoff  cd0.05  steps size-aware 2000-3000
logo   (Z)    : ai-toolkit linear32 + conv16  (standar)
```
Routing otomatis di `scripts/image_trainer.py` (`SDXL_NETWORK_BY_CATEGORY` + `detect_image_category`).
**Endpoint commit-hash → update ke HEAD `jalur-c-dethrone` sebelum submit (manual, oleh operator).**
