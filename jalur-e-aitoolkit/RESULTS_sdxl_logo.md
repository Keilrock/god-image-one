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

---

## FASE B2 — HEAD-TO-HEAD H1 vs H2 ⚠️ REPRODUKSI GAGAL (2026-06-27)

### Throughput gate
- H1 (DoRA) gate 20 step: **~1.94 s/it**. Full 540 step ≈ ~17 min/run. SDXL cepet. (H2 plain ~1.74 s/it.)

### Hasil (test 3 img, master_seed 42, weighted 0.25*text+0.75*no_text)
| ckpt | H1 (DoRA+conv) | H2 (boss plain) |
|------|----------------|------------------|
| epoch 25 | **0.05569** (best H1) | **0.05567** (best H2) |
| epoch 30 | 0.05607 | 0.05782 |
| epoch 35 | 0.05834 | 0.06149 |
| epoch 40 | 0.05828 | 0.06041 |
| last (ep45) | 0.05927 | 0.06516 |
| **BOSS #6 (sanity)** | **0.04652** | (target) |

### 🔴 VERDICT: GAGAL — H2 GAK reproduce boss, H1≈H2 (network BUKAN lever)
1. **H2 (replikasi boss plain) GAGAL reproduce 0.04653** → best 0.05567 (epoch25), last 0.06516. Meleset +20-40%.
2. **H1 ≈ H2 di best (0.05569 vs 0.05567)** → DoRA+conv vs plain **GAK ada beda** signifikan. Network BUKAN lever yg bikin boss menang. (Fase-A hipotesis "DoRA mudharat" → ternyata network gak ngaruh; dua-duanya sama-sama jelek.)
3. **Dua-duanya OVERFIT**: best epoch 25, makin naik abis itu. **text loss meledak** (0.074→0.103) — lora ngancurin prompt-following. Boss text cuma 0.048.
4. Pipeline VALID (sanity boss 0.04652 persis) → angka kita bener, masalah di TRAINING kita.

### 🔬 Diagnosis (kenapa kita overfit, boss enggak?) — HIPOTESIS, belum diuji
Recipe yg kita rekonstruksi (person bucket "s": prodigy d_coef1.1, lr0.95, 45 epoch, batch8, repeats5)
overtrains. Boss "last" = 0.04653 (gak overfit). Kandidat akar masalah:
- **A. Boss "last" ke-cut window LEBIH AWAL** (SDXL logo window 0.5-1.0h) → boss "last" = checkpoint zona bagus (~epoch 10-20), BUKAN 45. Kita train penuh 45 → overfit. (boss HF ckpt: last-000005..040+last — bisa jadi last=epoch≤40 hasil cut.) Tapi: bahkan epoch25 kita (0.0557) > boss 0.0465, jadi cut-timing aja gak cukup jelasin.
- **B. Bucket/LR salah rekonstruksi**: mungkin #6 bukan bucket "s" (18 img), atau prodigy d_coef ketinggian → LR overshoot → overfit + text rusak. (data[anima] lrs kita==boss, TAPI size_key mungkin beda kalau img count efektif beda.)
- **C. Repeats/effective-steps mismatch**: 18×5/8 = ~11 step/epoch × 45 = ~495 step. Kalau boss efektif < itu → kita over-train.
- **D. prodigy konvergen beda** vs setup boss (seed/init) — tapi gap terlalu besar buat cuma variance.

### Lever boss BELUM pakai (referensi edge — TAPI diagnosa dulu, jangan tebak):
noise_offset (none), caption_dropout (0), ema (none), min_snr beda. NAMUN: masalah utama = OVERFIT + text rusak,
bukan kurang-regularisasi doang. Prioritas = cari kenapa text-following ancur (kemungkinan LR prodigy ketinggian / over-train), BUKAN tambah lever.

### STATUS B2: STOP — reproduksi gagal, network bukan lever. Butuh diagnosa akar (window-cut? LR? bucket?) sebelum lanjut. NUNGGU putusan Wen.

---

## FASE B3 — DIAGNOSTIK TRAJEKTORI BOSS (2026-06-27) → VERDICT: RECIPE-WRONG

Eval checkpoint boss #6 (HF) + checkpoint AWAL kita, apple-to-apple per epoch.

### Trajektori boss #6 vs kita (weighted, test 3 img master 42)
| epoch | BOSS #6 | OUR H2 (plain=boss recipe) | OUR H1 (DoRA+conv) |
|-------|---------|----------------------------|--------------------|
| 5  | 0.06398 | 0.06310 | 0.06341 |
| 10 | **0.04556** | 0.06129 | 0.05933 |
| 15 | — | 0.05795 | 0.05582 |
| 20 | 0.04650 | 0.05578 | ~0.0556 |
| 25 | — | 0.05567 | 0.05569 |
| 30 | 0.04940 | 0.05782 | 0.05607 |
| 40 | 0.05528 | 0.06041 | 0.05828 |
| **last (submit)** | **0.04652** | 0.06516 | 0.05927 |

### 🔑 Temuan diagnostik
1. **Boss OPTIMAL = epoch 10 (0.04556)**, lalu boss JUGA overfit (20→0.0465, 30→0.0494, 40→0.0553).
2. **Boss "last.safetensors" (0.04652) = checkpoint TERPILIH miner (~epoch 10-20), BUKAN epoch 45.** Boss submit best-early, bukan final. (boss epoch40=0.0553 > boss last=0.0465 → "last" pasti checkpoint awal yg dipilih.)
3. 🔴 **SAME-EPOCH: epoch 10 boss 0.04556 vs kita 0.0613 (H2) / 0.0593 (H1).** Di epoch 5 kita SAMA boss (~0.063), tapi boss LOMPAT konvergen epoch 5→10 (0.064→0.0456), kita MANDEK (0.063→0.061).
4. **H1 ≈ H2 di semua epoch** → network (DoRA/conv) FINAL confirmed IRRELEVANT.
5. Config kita jalan bener (log: prodigy d_coef1.1, 90 img, 12 batch/epoch, repeats5, safeguard_warmup) → recipe rekonstruksi dieksekusi faithful.

### VERDICT DIAGNOSIS: **RECIPE-WRONG** (bukan window-cut, bukan checkpoint-selection)
- ❌ Window-cut: boss punya ladder epoch 5-40 penuh + overfit sendiri → BUKAN ke-cut paksa.
- ❌ Checkpoint-selection doang: kalau cuma itu, epoch-sama harusnya match. TAPI epoch 10 boss 0.0456 vs kita 0.061 → beda fundamental.
- ✅ **RECIPE BEDA**: recipe boss yg sebenernya bikin konvergen CEPAT & DALAM (epoch10=0.0456). Recipe kita (rekonstruksi dari lrs GitHub) konvergen LAMBAT & DANGKAL (best ~0.0556). Boss's true LR/optimizer dynamics ≠ rekonstruksi kita — metadata boss DI-STRIP, jadi optimizer/LR/step asli boss GAK keliatan; rekonstruksi dari lrs jelas GAGAL reproduce.

### Akar masalah (kandidat — buat keputusan Wen, JANGAN auto-train):
- **Konvergensi 5→10 boss jauh lebih agresif** → effective LR boss di window itu LEBIH TINGGI. prodigy kita (safeguard_warmup + d_coef1.1) ramp kelambatan ATAU bukan yg boss pakai.
- lrs "s" bucket (18img) mungkin BUKAN yg boss pakai buat #6 (kalau img-count efektif beda → bucket beda → LR/batch/epoch beda).
- Metadata stripped → optimizer/LR/step boss = UNKNOWN. Rekonstruksi = inferensi, terbukti meleset.

### Next (opsi, nunggu Wen):
- A. Sweep prodigy d_coef lebih tinggi / warmup off → kejar konvergensi epoch-10 boss.
- B. Cek bucket lain (xs: d_coef1.2 batch4 48ep) — mungkin #6 ke bucket beda.
- C. Pilih best-early checkpoint kita (epoch ~25, 0.0557) = submission "aman" TAPI masih KALAH boss 0.0465 (−0%, kalah 20%). Belum cukup.

## STATUS B3: SELESAI — RECIPE-WRONG confirmed. Best kita 0.0557 vs boss 0.0465 (kalah ~20%). Network bukan lever. STOP, nunggu keputusan lever recipe (prodigy/bucket).

---

## FASE B4 — CEK SOURCE BOSS LEBIH DALAM → 🎯 AKAR KETEMU: caption_dropout_rate

User minta cek source GitHub boss lengkap sebelum nebak lever. Template SDXL + resolve penuh.

### Temuan
1. `base_diffusion_sdxl_person.toml` boss == kita **IDENTIK** (byte-for-byte). Template bukan masalah.
2. Resolve logo #6 = template person + lrs[anima]["s"] bucket + network[228] + **line 307: `config["caption_dropout_rate"] = 0.1`**.
3. 🔴 **caption_dropout_rate di-OVERRIDE jadi 0.1 di AKHIR create_config (line 307)** — override template (yg =0). Gue kelewat karena build config MANUAL (pakai nilai template 0), bukan lewat pipeline image_trainer.py.

### DIFF config efektif boss vs yg gue jalanin (H2)
| key | BOSS-effective | GUE (ran) |
|-----|----------------|-----------|
| optimizer / d_coef / LR / epochs / batch / min_snr / network / seed / scheduler | (semua) | **IDENTIK** |
| **caption_dropout_rate** | **0.1** | **0** 🔴 SATU-SATUNYA BEDA |

### Kenapa ini AKAR-nya (cocok 100% sama gejala)
- **cd=0 (gue)**: model overfit ke 18 caption training → generalisasi ke test caption JELEK → **text loss 0.088**, konvergen lambat & dangkal (best 0.0557).
- **cd=0.1 (boss)**: 10% step caption di-drop → regularisasi text-conditioning → generalisasi bagus → **text loss 0.050**, konvergen cepat & dalam (epoch10=0.0456).
- Gejala kita PERSIS = overfit caption (text-following ancur). caption_dropout = lever yg nyegah itu.

### VERDICT: bukan lever tebakan — ini SETTING BOSS ASLI dari source yg gue lewat.
Recipe boss-verbatim yg BENER = H2 + **caption_dropout_rate=0.1**. Belum di-train (per aturan STOP).
Network (DoRA/conv) tetep irrelevant — fix = cd, bukan network.

### NEXT (rekomендasi, nunggu Wen): 
Re-train H2-corrected (plain LoRA 32/32 + **cd=0.1**, sisanya sama) → target reproduce boss ~0.0456 di epoch ~10.
Kalau reproduce → baru cari edge buat NYALIP (mis. cd sweep, atau best-early-checkpoint pick kayak boss).

## STATUS B4: AKAR KETEMU (caption_dropout 0.1, dari source). STOP, nunggu go re-train H2-corrected.

---

## FASE B5 — AUDIT PIPELINE ASLI (FIX: stop config manual) (2026-06-27)

User benar: eksperimen B2 pakai config MANUAL, BUKAN pipeline asli repo. Bahaya — turnamen jalanin pipeline asli,
config manual gak nyerminin + bug pipeline gak ke-detect. Audit semua runtime override + konfirmasi pipeline asli.

### ✅ KONFIRMASI: pipeline asli dipakai (panggil `create_config()` beneran via image trainer-deps)
REAL config pipeline jalur-e buat logo #6 (trigger `Brandmark_Essentials`, 18 img → bucket "s", category auto="logo"):
```
network_module = lycoris.kohya  (DoRA)
network_args = [conv_dim=4, conv_alpha=4, algo=lora, dora_wd=True, dropout=0]
network_dim/alpha = 32/32
caption_dropout_rate = 0.05      <-- pipeline runtime override (line 499), BUKAN 0
optimizer=prodigy d_coef1.1, lr0.95, 45ep, batch8, min_snr6, constant, seed 2951032222
```

### 🔴 RUNTIME OVERRIDE / PERILAKU PIPELINE yg config MANUAL KELEWAT
| # | hal | manual gue | pipeline ASLI kita (line) | pipeline BOSS |
|---|-----|-----------|---------------------------|---------------|
| 1 | **caption_dropout_rate** | **0** ❌ | **0.05** (img_trainer:499, _cd_default) | **0.1** (img_trainer:307) |
| 2 | **LLaVA auto-caption** | ❌ pakai .txt mentah | ✅ auto_caption_dataset: `.txt + ", " + llava_caption` (butuh /opt/models/llava) | ✅ sama (llava) |
| 3 | **network category routing** | hardcode H1/H2 | ✅ auto-detect "logo"→DoRA+conv (485-488) | plain LoRA (228, person map) |
| 4 | lrs "s" bucket (LR/opt/ep/batch/min_snr) | ✅ ke-include | ✅ (340/357) | ✅ sama |
| 5 | template person fields | ✅ | ✅ | ✅ identik |
| 6 | repeats 5 / folder "5_lora style" | ✅ | ✅ | ✅ |

→ **3 hal kelewat di manual: (1) caption_dropout, (2) LLaVA caption augmentation, (3) network auto-routing.**
Yang PALING ngefek ke gejala kita (text-following ancur): **#1 cd + #2 llava-caption** (dua-duanya soal caption/text).

### 🔴 TEMUAN STRATEGIS: pipeline ASLI KITA ≠ BOSS (2 deviasi dethrone)
- **caption_dropout: kita 0.05 vs boss 0.1** (dethrone ubah dari 0.1→0.05).
- **network: kita DoRA+conv vs boss plain LoRA** (dethrone tambah category routing).
- → Walau pakai pipeline asli kita, GAK reproduce boss (recipe kita udah dimodif dethrone menjauh dari boss).
- Catatan: B2 manual (cd=0) makin jauh lagi. Pipeline asli (cd=0.05) lebih deket boss tapi belum sama (0.1).

### NEXT (nunggu Wen) — semua VIA PIPELINE ASLI, no manual:
1. **Re-run via pipeline asli** (build trainer image + llava) → recipe jalur-e sebenarnya (DoRA+cd0.05+llava-caption). Lihat landing vs boss.
2. Kalau masih kalah: A/B lewat env override pipeline (DETHRONE_SWEEP_CD) atau ubah `_cd_default`/network di image_trainer.py → samain/ngalahin boss (cd 0.1, plain). TANPA config manual.
3. Butuh: llava 13GB (auto_caption), trigger word "Brandmark_Essentials", window #6.

## STATUS B5: AUDIT SELESAI. 3 runtime miss (cd/llava/network) teridentifikasi, pipeline asli dikonfirmasi (cd=0.05/DoRA). Pipeline kita ≠ boss (deviasi dethrone). STOP, nunggu putusan: re-run pipeline asli as-is atau fix dulu.
