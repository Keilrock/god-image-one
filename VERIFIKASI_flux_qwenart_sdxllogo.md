# VERIFIKASI Flux / Qwen-art / SDXL-logo via create_config (CPU, no GPU, no train)

> Tujuan: mastiin recipe 3 task ini BENERAN ke-generate bener lewat pipeline (bukan "katanya kebawa").
> Cara: baca template + jalanin `create_config()` ASLI lewat entrypoint (CPU), banding output vs recipe MENANG.
> Branch jalur-e-aitoolkit. Patokan: GIT_HISTORY_MAP.md (task 2/4/5). Lapor ANGKA, bukan kesimpulan.
> Harness: drive `image_trainer.main()` asli via CLI args, mock auto_caption(LLaVA)+run_training (stop sebelum train),
> lycoris-lora di-install biar DoRA gak fallback. Dataset asli: f9459ba8 (art), 7489f54c (logo).

---

## VERIFIKASI 1 — FLUX 🟢 (1 deviasi kecil)

`create_config` model `mhnakif/fluxunchained-dev` → `hash = 6445f395…` = key flux.json `{}` → resolve template `base_diffusion_flux.toml` verbatim. (flux.json `default = {}`, revert 738f4ae terkonfirmasi.)

| field | pipeline generate | recipe juara (0.03507) | cocok? |
|---|---|---|---|
| network_module | networks.lora_flux | lora_flux | ✅ |
| network_dim/alpha | **128 / 64** | 128/64 | ✅ |
| optimizer_type | **Lion** | Lion | ✅ |
| guidance_scale | **85.0** | 85.0 | ✅ |
| discrete_flow_shift | **3.1582** | 3.1582 | ✅ |
| unet_lr / te_lr | **8e-5 / [8e-6, 8e-6]** | 8e-5 / [8e-6,8e-6] | ✅ |
| max_train_steps | **250** | 250 | ✅ |
| network_args | train_t5xxl=True (+ double/single all) | train_t5xxl | ✅ |
| timestep_sampling | sigmoid | sigmoid | ✅ |
| seed | 2 | 2 | ✅ |
| **caption_dropout_rate** | **0.05** | **0.1** | 🟡 **BEDA** |

🟡 **FLAG:** semua lever utama PERSIS juara, **kecuali caption_dropout: pipeline 0.05 vs run-juara 0.1**.
Bukti run-juara cd 0.1: `jalur-e-aitoolkit/flux_src_ref/config_full.toml:57` & `config_gate_used.toml:57` = `caption_dropout_rate = 0.1`; `RECIPE_flux_task4.md:158`. Pipeline pakai `_cd_default=0.05` (flux gak masuk cabang `if model_type=='sdxl'`).
Dampak kemungkinan kecil (250 step, dataset kecil) TAPI secara teknis beda dari run yang menang. Kalau mau persis: set flux cd → 0.1 (1 baris). **Keputusan user.**
🟢 Leftover rank32/adamw/1000 **TIDAK muncul** — resolve ke template juara.

---

## VERIFIKASI 2 — QWEN-ART 🟢 (low-risk, bukan person recipe)

`create_config` model Jib-Mix, **NO trigger**, dataset art 13 img.

| field | generate | banding | status |
|---|---|---|---|
| category | **art** (no trigger → fail-safe) | — | ✅ |
| use_ema | **True** | art = EMA-ON | ✅ |
| steps | **2600** (size-aware 13×200) | eksperimen menang **2500** | 🟢 ≈ (+100, ~4%) |
| caption_dropout | 0.05 | template 0.05 | ✅ |
| person recipe? | steps≠156 (12×13), EMA on | guard person | ✅ **BUKAN person** |

🟢 **Steps 2600 vs 2500 = aman.** Dua-duanya di-**cut window** (art 1.0–1.5h) ke ~1000–1250 step saat runtime
(kill by `--hours-to-complete`, BUKAN di create_config). Final checkpoint ≈ sama (winner best ~step 1000).
**BUKAN** under/overtrain — selisih 100 step pada angka tulisan yang sama-sama ke-cap window.
Angka step PERSIS yang dilatih gantung kecepatan GPU/window → konfirmasi final pas re-validate GPU.
🟢 Guard person kebukti: art **TIDAK** kena EMA-off / 12×img.

---

## VERIFIKASI 3 — SDXL LOGO 🟢 recipe MENANG kebawa — 🔴 epoch beda dari dugaan

`create_config` sdxl base, trigger "Brandmark_Essentials", dataset logo 18 img.
category ke-detect = **logo** ✅. lycoris ke-install → DoRA aktif (gak fallback plain-64). template = person.toml (Styles:[] → is_style False).

| field | pipeline generate | manual SALAH (kalah 0.0557) | status |
|---|---|---|---|
| category | **logo** | — | ✅ |
| network_module | **lycoris.kohya** (DoRA) | H1 DoRA / H2 plain | ✅ recipe jalur-c |
| network_dim/alpha | **32 / 32** | 32/32 | ✅ |
| network_args | **conv_dim=4, conv_alpha=4, algo=lora, dora_wd=True, dropout=0** | — | ✅ jalur-c |
| **caption_dropout** | **0.05** | **0** (overfit → kalah) | ✅ **BUKAN manual** |
| optimizer d_coef | **1.0** | 1.1 | ✅ bukan manual |

→ 🟢 **Pipeline generate recipe MENANG jalur-c (DoRA + conv4/4 + cd0.05), BUKAN manual kalah (cd0 / d_coef1.1).**
Manual salah cuma di `sdxl_logo_NOTES_flawed_manual/`, gak nyangkut pipeline. **Inti verifikasi 3 terpenuhi.**

### 🔴 KOREKSI ASUMSI PROMPT (temuan penting)
epoch/min_snr/batch **BUKAN** dari template person.toml (ep25/min_snr5). Di-**OVERRIDE lrs `person_config.json` size-bucket**.
18 img → bucket **'s'** (11–20 img) →
- `max_train_epochs = 60` (BUKAN template 25, BUKAN manual 45)
- `min_snr_gamma = 6` (BUKAN template 5)
- `train_batch_size = 6`, `prior_loss_weight = 0.78`, `optimizer d_coef = 1.0`

Jadi recipe SDXL-logo asli pipeline = **DoRA32/conv4/cd0.05 (routing) + lrs-'s'-bucket (ep60/min_snr6/batch6/d_coef1.0)**.

⚠️ **Jujur:** network+cd = recipe MENANG jalur-c terkonfirmasi. TAPI **epoch gantung jumlah gambar** (size-bucket):
18img→ep60; task logo lain dengan img beda → bucket beda → epoch beda. **Belum bisa dipastiin** ep60 == angka persis
yang menang −16.9% di T5 jalur-c (T5 punya img count sendiri → mungkin bucket lain) tanpa sweep-log jalur-c.
Yang pasti: pipeline deterministik ngeluarin recipe DoRA+cd0.05 size-bucketed = desain jalur-c, **bukan** manual yang kalah.

---

## RINGKASAN VONIS

| Task | Pipeline = recipe menang? | Flag |
|---|---|---|
| **Flux** | 🟢 ya, kecuali **cd 0.05 vs 0.1** | 🟡 set flux cd=0.1 kalau mau persis |
| **Qwen-art** | 🟢 ya (EMA-on, steps 2600≈2500 window-cut) | low-risk |
| **SDXL-logo** | 🟢 DoRA+cd0.05 (menang), BUKAN manual | 🔴 epoch dari lrs-bucket (ep60), bukan template ep25; size-dependent |

### Catatan teknis
- Harness CPU (entrypoint + stub fiber/transformers, lycoris-lora installed) di `/ephemeral/f2test/` (ephemeral, gak di-commit).
- LLaVA di-mock (caption mentah). Untuk SDXL/qwen, detector/routing baca caption — di produksi LLaVA append dulu (low-risk, lihat analisa caveat sebelumnya).
- Re-validate GPU (1 run/task) tetap perlu untuk konfirmasi skor akhir + step final (window-cut) + caption LLaVA-augmented.
