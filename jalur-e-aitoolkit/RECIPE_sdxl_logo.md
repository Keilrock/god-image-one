# SDXL LOGO — FASE A: BONGKAR BOSS + BANDING (no GPU, analisis doang)

Front terakhir 5/6. Logo SDXL = satu-satunya yg masih KALAH (−0.5% T5 11 Jun). Target: NGALAHIN boss.

## 1. Lokasi recipe logo KITA
- **Branch: `jalur-e-aitoolkit`** (asal `jalur-c-dethrone`, kebawa ke jalur-e — confirmed ada di dua-duanya).
- File: `scripts/image_trainer.py` (routing per-kategori `SDXL_NETWORK_BY_CATEGORY`) + `scripts/core/config/base_diffusion_sdxl*.toml` (template) + `scripts/lrs/{person,style}_config.json` (LR/step override per-base).
- **Recipe logo kita (network):** `lycoris.kohya` **DoRA**, dim32/alpha32, **conv_dim4/conv_alpha4**, algo=lora, **dora_wd=True**, dropout=0, **NO loraplus_lr_ratio**.
- LR/opt/step: dari lrs (is_style→style_config, else person_config) per size-bucket.

## 2. Recipe BOSS logo — DUA sumber cross-check

### 2a. HF METADATA model logo boss #6 18-Jun (GROUND TRUTH — yg beneran ke-train)
Repo: `...-5b5b9689-...-5GU4Xkd3`, checkpoint `last.safetensors`.
- ⚠️ Metadata **DI-STRIP** (17 keys, kayak Flux): no ss_learning_rate/optimizer/network_args/steps.
- Yg survive: **ss_network_module = `networks.lora` (PLAIN LoRA, BUKAN DoRA/lycoris!)**, ss_network_dim **32**, ss_network_alpha **32**, base sdxl_base_v1-0.
- 🔴 **Tensor: conv = 0** (2958 tensor: 2166 unet + 792 te, NOL conv) → **boss logo TIDAK pakai conv**.

### 2b. SOURCE GitHub position-1 18-Jun (niat config) — cocok sama metadata?
- `image_trainer.py`: SDXL network dari `config_mapping[network_config_{person/style}[base]]`, module tetap `networks.lora`.
- anima-pencil (base #6): `network_config_person`→**228**, `network_config_style`→235.
  - config_mapping[228] = dim32/alpha32/**network_args=[] (NO conv)** ✅ COCOK metadata.
  - config_mapping[235] = dim32/alpha32/conv4 (ini kalau is_style).
- → Metadata (32/32/no-conv) = **mapping 228 = PERSON route**. Artinya boss `detect_styles_in_prompts(logo)` = **False → person template + person_config + mapping 228**.
- **lrs anima-pencil (boss person_config):** prodigy, unet_lr/te_lr ~0.9, lr_scheduler constant, min_snr_gamma 6, size-bucket epochs (xs48/s45/m36...). default boss = `{}`.
- ✅ **Cocok: config GitHub == metadata HF.** Gak ada hijack lrs (beda dari kasus flux.json). Boss logo = plain LoRA 32/32 no-conv + person-prodigy, konsisten.

## 3. 🔴 DIFF SIDE-BY-SIDE (logo #6 anima-pencil)

| lever | LOGO KITA | LOGO BOSS (menang) | STYLE KITA (T4 MENANG) |
|-------|-----------|--------------------|-----------------------|
| **network_module** | **lycoris.kohya (DoRA)** | **networks.lora (plain)** | lycoris.kohya (DoRA) |
| dim / alpha | 32 / 32 | 32 / 32 | 32 / 32 |
| **conv** | **conv4/conv_alpha4** | **TIDAK ADA** | conv4/conv_alpha4 |
| dora_wd | True | — (no DoRA) | True |
| **loraplus_lr_ratio** | **TIDAK ADA** | TIDAK ADA | **16** ✅ |
| dropout | 0 | null (=0) | 0 |
| optimizer | prodigy | prodigy | adamw |
| unet_lr / te_lr | ~0.9 / ~0.9 (prodigy) | ~0.9 / ~0.9 (prodigy) | 2.2e-5 / 1e-6 (adamw) |
| lr_scheduler | constant | constant | constant_with_warmup/cosine |
| min_snr_gamma | 6 | 6 | 6 |
| steps/epochs | person bucket (size) | person bucket (size) | style bucket |
| base | anima-pencil | anima-pencil | (T4 fluently-xl) |

🔑 **Buat anima-pencil logo: LR/optimizer/scheduler/steps/min_snr KITA == BOSS PERSIS** (data[anima] di lrs IDENTIK ours==boss, prodigy person-bucket). **Satu-satunya beda = NETWORK:** kita DoRA+conv, boss plain LoRA no-conv.

## 4. Analisa lever

### Lever yg BIKIN BEDA logo kita vs boss:
1. 🔴 **DoRA vs plain LoRA**: kita lycoris DoRA, boss `networks.lora` plain. Kita NAMBAH kompleksitas (DoRA decompose) yg boss GAK pakai — dan kita cuma setara challenger (KALAH −0.5%), boss plain LoRA MENANG. → **DoRA kemungkinan GAK bantu logo (malah noise/over-param)**.
2. 🔴 **conv4 vs no-conv**: kita train conv, boss enggak. Logo = render TEKS + bentuk vektor; konten frekuensi-tinggi/garis tajam lebih ke attention, bukan conv spatial. Boss menang TANPA conv → **conv kemungkinan buang kapasitas**.

### Banding STYLE-kita-MENANG vs LOGO-kita-KALAH (apa beda?):
- Style (MENANG T4): DoRA + conv4 + **loraplus_lr_ratio=16**.
- Logo (KALAH): DoRA + conv4, **TANPA loraplus**.
- → Rekap betul: **logo kita gak ada LoRA+, style ada.** loraplus naikin LR matrix-B (detail high-freq) → bantu style. TAPI: boss MENANG logo **tanpa loraplus DAN tanpa DoRA** (plain LoRA). Jadi loraplus belum tentu kunci — boss buktiin SIMPLE menang.

## 5. Hipotesis lever FLIP (prioritas)

**H1 (PALING kuat, lowest-risk) — REPLIKASI BOSS VERBATIM: logo → plain `networks.lora` 32/32 NO-conv.**
LR/opt/step udah identik boss → tinggal samain NETWORK = recipe boss PERSIS. Preseden Flux: replikasi verbatim + non-determinism run-to-run NYALIP juara. −0.5% itu tipis (hinge), 1 run bagus bisa flip. Risiko paling kecil, ground-truth-proven.

**H2 (edge candidate) — plain LoRA 32/32 + `loraplus_lr_ratio=16` (graft lever pemenang STYLE ke base pemenang BOSS).**
Ambil base proven boss (plain LoRA no-conv) + tambahin satu-satunya lever yg bikin STYLE kita menang (loraplus16). loraplus boost LR-B → tajamkan render teks logo. Edge NYATA potensial tanpa kompleksitas DoRA. Worth A/B vs H1.

**H3 (turunin prioritas) — naikin rank buat teks (64/64).**
Argumen logo butuh presisi teks. TAPI boss menang di rank 32 → rank bukan bottleneck. Skip dulu.

**H4 (buang) — pertahanin DoRA+conv current.** Ground truth: boss plain-no-conv MENANG, DoRA+conv kita KALAH. DoRA & conv kemungkinan MUDHARAT buat logo. Jangan diterusin.

### Rekomendasi A/B Fase B (nunggu Wen):
- **Baseline = H1** (boss verbatim plain LoRA no-conv) → target replikasi ~0.0410-0.0411, flip via run bagus.
- **A/B = H2** (plain + loraplus16) → cek apakah loraplus kasih edge nyata di logo (kayak di style).
- Dua-duanya buang DoRA+conv (terbukti gak bantu). Eval pakai pipeline validator (image-evaluator) yg udah tervalidasi di Flux.

## Catatan
- 11 Jun T5 (protovision) position-1 BELUM di-extract terpisah; pola boss logo (plain LoRA) konsisten dari #6 ground truth. Bisa cross-check nanti kalau perlu.
- Metadata boss stripped → LR/opt/step dari GitHub source (cocok, no hijack).

## STATUS: FASE A SELESAI — STOP. No edit/train. Nunggu putusan Wen titik awal + A/B.
