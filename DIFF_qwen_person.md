# DIFF Qwen person — Template pipeline (A) vs Winner 5FW2 (B) vs Eksperimen EMA-off kita (C)

> Audit config-only (NO GPU, NO train). Tujuan: tahu SEMUA lever yang harus ditimpa ke pipeline
> biar recipe pemenang Qwen person (EMA-off, skor **0.06825**) kebawa saat validator jalanin entrypoint.
> Patokan: GIT_HISTORY_MAP.md F2. Submission branch jalur-e-aitoolkit HEAD.

Sumber:
- **(A)** `scripts/core/config/base_diffusion_qwen_image.yaml` — template HEAD yang di-resolve `create_config()`.
- **(B)** `jalur-d-perjuangan/winner-recipes/winner_qwen_aitoolkit_config.yaml` — recipe boss 5FW2Eaae (kebongkar jalur-d).
- **(C)** `jalur-e-aitoolkit/configs/ema_off_config.yaml` — config eksperimen kita yang ngasih 0.06825.

🔴 **PENTING:** kolom (A) bukan cuma isi file template. `create_config()` (image_trainer.py, jalur ai-toolkit
L275–318) **menimpa `steps` lewat size-aware** sebelum train. Jadi (A) ditulis dua: *file* dan *EFEKTIF* (setelah create_config).

---

## TABEL DIFF 3 ARAH (per lever)

| Lever | (A) Template HEAD (efektif via create_config) | (B) Winner 5FW2 | (C) EMA-off kita (0.06825) | A vs C |
|---|---|---|---|---|
| network.type | lora | lora | lora | ✅ sama |
| network.linear (rank) | 128 | 128 | 128 | ✅ |
| network.linear_alpha | 128 | 128 | 128 | ✅ |
| **ema_config.use_ema** | **true** | true | **false** | 🔴 **BEDA** |
| ema_config.ema_decay | 0.995 | 0.995 | 0.995 (moot, EMA off) | ✅ |
| **steps (file)** | **2500** | 108 | 108 | 🔴 |
| **steps (EFEKTIF setelah create_config)** | **2000** (floor; dataset 9img→1800→max(2000,..)=2000; bisa s/d 3000 kalau ≥15img). Window-cap ~1250 by time-kill. | **108** | **108** | 🔴🔴 **PALING KRITIS** |
| lr | 0.0001 | 0.0001 | 0.0001 | ✅ |
| optimizer | adamw8bit | adamw8bit | adamw8bit | ✅ |
| optimizer_params.weight_decay | 1e-5 | 1e-5 | 1e-5 | ✅ |
| do_cfg | true | true | true | ✅ |
| cfg_scale | 6.0 | 6.0 | 6.0 | ✅ |
| **resolution (multi-res)** | **[512,768,1024] ADA** | [512,768,1024] | [512,768,1024] | ✅ **ADA di template** |
| timestep_type | **weighted (ADA)** | weighted | weighted | ✅ |
| noise_scheduler | flowmatch | flowmatch | flowmatch | ✅ |
| dtype / save dtype | bf16 / bf16 | bf16 / bf16 | bf16 / bf16 | ✅ |
| quantize / qtype | true / float8 | float8 | float8 | ✅ |
| quantize_te | true | true | true | ✅ |
| low_vram | true | true | true | ✅ |
| **caption_dropout_rate** | **0.05** (dataset block; create_config TIDAK timpa utk ai-toolkit) | *tidak ada* (default ai-toolkit) | *tidak ada* (default ai-toolkit) | 🟡 **BEDA** (lihat catatan) |
| batch_size | 1 | 1 | 1 | ✅ |
| gradient_accumulation | 1 | 1 | 1 | ✅ |
| gradient_checkpointing | true | true | true | ✅ |
| train_unet / train_text_encoder | true / false | true / false | true / false | ✅ |
| cache_text_embeddings | true | true | true | ✅ |
| cache_latents_to_disk | true | true | true | ✅ |
| trigger_word | dari arg `--trigger-word` (create_config set `process['trigger_word']`) | "David Miller" | "David Miller" | ✅ (via arg) |
| save_every | 250 | 250 | 36 | 🟡 kosmetik (granularitas checkpoint, bukan lever training) |
| save_format | diffusers | diffusers | diffusers | ✅ |

**Kesimpulan kecocokan:** dari ~25 lever, **A vs C identik di SEMUA kecuali 3**: `steps` (2000 vs 108), `use_ema` (true vs false), `caption_dropout` (0.05 vs default). Sisanya (network/lr/opt/cfg/res/timestep/wd/fp8/trigger) sudah cocok di template.

---

## YANG BEDA antara Template HEAD (A) vs recipe MENANG (C) — ditegasin

### 🔴🔴 1. STEPS — beda paling fatal (BUKAN cuma EMA)
- Template ditulis 2500, TAPI `create_config()` **menimpa** lewat size-aware:
  `compute_aitoolkit_steps(dataset_size, per_image=200, min_steps=2000, max_steps=3000)` = `max(2000, min(dataset_size*200, 3000))`.
- **Floor-nya 2000.** Dataset Qwen person ~9 img → 1800 → di-floor jadi **2000**. Tidak ada ukuran dataset yang bisa hasilin 108.
  → Pipeline **overtrain ~18× lipat** (2000 vs 108; bahkan setelah window-cap ~1250, tetap ~12×). Ini sebab kuat kenapa pipeline ≈ 0.084 (KALAH), bukan 0.068.
- ⚠️ **Edit template `steps:108` TIDAK cukup** — size-aware di image_trainer.py bakal nimpa balik ke 2000. **Wajib ubah KODE** (guard/skip size-aware utk Qwen person) supaya 108/pass-per-img ~4 kebawa.

### 🔴 2. EMA — template ON, butuh OFF
- Template `use_ema: true` (warisan boss c65c405, **tak pernah diubah**). Pemenang kita `use_ema: false`.
- EMA-on di low-step person nyungsep magnitude weight (Δweight ~82 vs 247) → identitas lemah. EMA-off = 0.06825 vs EMA-on 0.0716.

### 🟡 3. CAPTION_DROPOUT — template paksa 0.05, eksperimen tanpa cd
- Template dataset `caption_dropout_rate: 0.05` (ditambah dethrone 4a9952b). Config (B) winner & (C) yang ngasih 0.06825 **tidak mencantumkan** cd → pakai default ai-toolkit.
- Artinya skor menang 0.06825 dihasilkan TANPA cd eksplisit 0.05. Kalau lewat pipeline, cd0.05 ikut kepasang → **belum tentu sama** dengan kondisi yang ngasih 0.06825.
- ⚠️ Perlu verifikasi: default caption_dropout ai-toolkit (kemungkinan 0). Komentar dethrone klaim "juara Qwen pakai 0.05" tapi extraction (B) tak menunjukkannya. Lever minor dibanding steps/EMA, tapi kalau mau replikasi PERSIS kondisi 0.06825, samakan dengan (C).

### ✅ Yang SUDAH benar di template (dijawab eksplisit)
- **MULTI-RES [512,768,1024]: ADA** di template ✅.
- **CFG (do_cfg true, cfg_scale 6.0): ADA** ✅.
- **timestep_type weighted: ADA** ✅.
- **weight_decay 1e-5: ADA** ✅.
- **fp8 (quantize float8 + quantize_te): ADA** ✅.
- network 128/128, lr1e-4, adamw8bit, flowmatch, trigger via arg: ADA ✅.

---

## LIST PERSIS yang harus ditimpa/ditambah biar `create_config` generate recipe MENANG

> ⚠️ JANGAN diterapkan dulu (prompt = STOP). Ini daftar keputusan.

1. **STEPS → 108 (atau pass-per-img ~4).** Tidak bisa via template doang. Butuh ubah KODE `image_trainer.py`:
   guard blok size-aware (L296–311) agar untuk **Qwen person** tidak nge-floor ke 2000 — set steps fixed ~108
   (atau formula pass-per-img rendah). **Ini perubahan terpenting.**
2. **`ema_config.use_ema: true → false`** di `base_diffusion_qwen_image.yaml`.
3. **caption_dropout:** putuskan — hapus dari template (samakan dgn C/B) ATAU konfirmasi default ai-toolkit
   == kondisi yang ngasih 0.06825. (verifikasi dulu).
4. (opsional) `save_every` lebih kecil supaya checkpoint best-early ke-capture.

---

## 🔴 CAVEAT KRITIS — fix WAJIB SCOPED ke Qwen PERSON, jangan global

Template Qwen dipakai SEMUA task Qwen. Kalau steps=108 & EMA-off di-set global, **rusak task lain:**

| Task Qwen | Butuh | Kalau kena fix person global |
|---|---|---|
| **Qwen person** (task ini) | steps ~108, **EMA-OFF** | ✅ benar |
| **Qwen-art (task3)** | ~1000 step (window-cut dari 2500), **EMA-ON** (keputusan: EMA-off KALAH 0.07344, pakai boss EMA-on 0.07292) | 🔴 RUSAK — EMA-off & 108 step bikin art kalah |
| **Z** (kalau lewat ai-toolkit) | steps 2000 | 🔴 RUSAK kalau floor diturunin global |

→ Pipeline tidak bisa bedain Qwen-person vs Qwen-art di `create_config` kecuali ditambah **detektor** (mirip
`is_person_dataset` di SDXL): Qwen-person PUNYA trigger ("David Miller"), Qwen-art TANPA trigger (style di prosa).
Bisa pakai sinyal `trigger_word` ada/tidak + person-caption detection untuk nge-scope steps=108 + EMA-off
HANYA ke person. **Ini keputusan desain — bukan diputus di audit ini.**

**Ringkas:** untuk bawa pemenang Qwen person ke pipeline butuh (1) ubah kode steps (size-aware floor 2000 → 108 scoped),
(2) EMA-off scoped, (3) keputusan caption_dropout — dan **ketiganya harus scoped ke person** biar tidak ngerusak
Qwen-art & Z. Template-edit doang TIDAK cukup karena size-aware override + template dipakai bareng task lain.
