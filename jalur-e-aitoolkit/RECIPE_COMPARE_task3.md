# FASE A — Task #3 Qwen-Jib-Mix (art/style) : BONGKAR + BANDING

Analisis doang (NO train, NO download base gede). Tujuan: tau bedanya recipe boss-art #3 vs recipe person kita.

## Fakta task #3
- task_id: `1887298a-dea2-47e2-b8bb-b315bd5d32fa` | model_type: qwen-image | window 1.25h
- base: `gradients-io-tournaments/Qwen-Image-Jib-Mix` (BEDA dari Qwen-Image murni)
- dataset: dreamlike_and_digital_art — train f9459ba8 (13 img), test 93e6a3fa (2 img)
- 🏆 BOSS 5GU4Xkd3 MENANG: 0.07292 | 5FW2Eaae (recipe person #2) KALAH art: 0.08696

## Tabel banding 3 recipe

| field | (a) BOSS #3 art `5GU4` ← menang art | (b) EMA-off person kita | (c) winner person `5FW2` #2 |
|-------|------|------|------|
| **base** | **Qwen-Image-Jib-Mix** | Qwen-Image | Qwen-Image |
| network | lora 128/128 | lora 128/128 | lora 128/128 |
| **steps** | **2500** | 108 | 108 |
| #img | 13 | 9 | 9 |
| **steps/img** | **~192** | 12 | 12 |
| **EMA** | **ON, decay 0.995** | **OFF** | ON, decay 0.995 |
| do_cfg | true | true | true |
| cfg_scale | 6.0 | 6.0 | 6.0 |
| timestep_type | weighted | weighted | weighted |
| lr | 1e-4 | 1e-4 | 1e-4 |
| optimizer | adamw8bit (wd 1e-5) | adamw8bit (wd 1e-5) | adamw8bit (wd 1e-5) |
| **trigger_word** | **(TIDAK ADA)** | David Miller | David Miller |
| resolution | 512/768/1024 | 512/768/1024 | 512/768/1024 |
| quantize | float8 + quantize_te | float8 + quantize_te | float8 + quantize_te |
| noise_scheduler | flowmatch | flowmatch | flowmatch |
| save_format | diffusers | diffusers | diffusers |

## Caption train set f9459ba8 (13 img) — KONFIRMASI format art/style
- **NO trigger word.** Style name dijahit di PROSA caption, 13/13 caption: "...in a **Dreamlike and Digital Art** style."
- Caption = prosa deskriptif panjang (33–45 kata): konten + style emphasis (tekstur, warna, komposisi).
- Contoh: *"A steampunk owl with intricate gear-work wings... in a Dreamlike and Digital Art style. Emphasize polished brass textures, glowing cog details..."*
- Beda total dari person #2 (trigger token "David Miller" + caption deskriptif portrait).
- ✅ Konfirmasi handoff note A: person/logo punya trigger, **art/style NULL (style dijahit caption)**.

## Yang bikin boss-art BEDA dari recipe person kita (lever kunci)
Struktur recipe **IDENTIK** di hampir semua field (rank 128, do_cfg 6.0, cfg_scale, timestep weighted, lr 1e-4, adamw8bit, multi-res, float8, flowmatch). Yang beda cuma **3 hal**:

1. 🔴 **STEPS: 2500 vs 108** (~16–23× lipat per-image). Ini lever DOMINAN art/style.
   Style internalization butuh training JAUH lebih lama dari identity 1-orang.
   (Inget: person kita 108 step udah cukup malah c36 terbaik — overfit cepat. Art kebalikan: butuh banyak step.)
2. **NO trigger_word** — style dijahit caption prosa, bukan token.
3. **base Qwen-Image-Jib-Mix** — base community art-tuned, bukan Qwen-Image vanilla. Kemungkinan udah bias ke estetika art → boss milih base yg "ngerti art".
4. **EMA: ON** (boss-art pakai EMA default-on, sama kayak winner person). Temuan EMA-off kita = person-specific; BELUM kebukti transfer ke art.

## Rekomendasi titik awal #3 (BUKAN keputusan final — nunggu Wen)
**Titik awal aman = recipe boss-art `5GU4` VERBATIM** (proven menang #3 @ 0.07292): steps 2500, EMA ON, no trigger, base Jib-Mix, sisanya identik recipe kita.

Alasan: semua lever (rank/cfg/timestep/lr/optimizer/multi-res) udah SAMA antara boss-art & recipe kita. Satu-satunya knob yg kita BUKTIIN berharga (EMA-off) justru yg boss-art GAK pakai. Jadi:
- **Baseline run** = boss-art verbatim (replikasi dulu, target ~0.07292).
- **A/B lever #1** = EMA-OFF (temuan person kita). Karena segalanya udah identik, EMA on/off jadi satu-satunya variabel bersih buat dicoba duluan.
- ⚠️ JANGAN pakai recipe EMA-off person (108 step) apa adanya → bakal massively undertrain art (butuh 2500). Step count = param art-specific paling kritis, jangan diutak-atik dulu.

## Hipotesis: lever apa yg matter buat art/style (vs person)
- **STEPS = lever utama**. Person (identity) overfit cepat → step kecil (c36 menang). Art (style) butuh banyak step (2500) buat internalize estetika konsisten. Kurva loss kemungkinan turun jauh lebih lambat.
- **EMA on/off — pertanyaan terbuka**: Person EMA-off menang krn EMA-on crush magnitude di step rendah (108). Di art 2500 step, dinamika beda — EMA averaging punya lebih banyak step "matang"; bisa jadi EMA-on malah bantu stabilitas style, ATAU EMA-off tetep naikin magnitude+koherensi. **Ini A/B paling worth dicoba.**
- **trigger = none** (caption prose) — fixed, ikut boss.
- **base Jib-Mix** = kemungkinan kontributor besar; base udah art-biased. Pakai yg sama kayak boss (jangan eksperimen base dulu).

## Estimasi base & feasibility
- `Qwen-Image-Jib-Mix`: total ~**78.1GB** full diffusers (transformer 5 shard ~41GB, TE 4 shard ~16.6GB, vae 0.25GB) ATAU **fp8 single-file 20.4GB** (`Jib_Mix_Qwen-Image_V4_E_fp8_e5m2_00001_.safetensors`).
- Footprint = SAMA persis kayak base person #2 (78GB) yg udah berhasil kita handle.
- Disk /ephemeral: 656GB free → **feasible**. Eval pakai fp8 (20GB), training butuh full diffusers (78GB).
- ⚠️ Training 2500 step (vs 108) → estimasi wall-clock ~23× train person (~5min) ≈ **~1.5–2 jam/run** (kasar; perlu ukur throughput real). Pertimbangan penting buat budget A/B.

## STATUS: FASE A SELESAI — STOP. Nunggu Wen putusin titik awal + strategi A/B sebelum Fase B (train).
