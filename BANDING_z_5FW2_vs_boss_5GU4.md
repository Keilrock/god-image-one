# BANDING recipe Z: pemenang 5FW2 vs BOSS 5GU4 (CPU, no GPU, no train)

> Tujuan: bongkar 2 model Z-Image 18-Jun (pemenang 5FW2 vs boss 5GU4) dari HF, banding head-to-head
> dari config.yaml + safetensors metadata. Cari lever apa yang bikin 5FW2 ngalahin boss (0.0401 vs 0.0414).
> Branch jalur-e-aitoolkit. Lanjutan KONFIRMASI_z_winner_5FW2.md. Gak ada yang diubah/commit.

Task sama: `956b14d3-7c4d-4d1a-8614-19d6bd025b2f` (Z-Image person "Evelyn Reed", **14 img**).
- 5FW2: `...-956b14d3-...-5FW2Eaae` (menang 0.0401)
- 5GU4: `...-956b14d3-...-5GU4Xkd3` (boss, kalah 0.0414)

---

## Tabel head-to-head
| field | 5FW2 (menang 0.0401) | boss 5GU4 (kalah 0.0414) | beda? |
|---|---|---|---|
| trainer | ai-toolkit 0.7.10 | ai-toolkit 0.7.10 | ✅ sama |
| network type | plain LoRA (0 dora_scale) | plain LoRA (0 dora_scale) | ✅ sama |
| rank/alpha | 32/32 | 32/32 | ✅ sama |
| conv/conv_alpha | 16/16 | 16/16 | ✅ sama |
| modul count | 240 | 240 | ✅ sama |
| lr | 1e-4 | 1e-4 | ✅ sama |
| optimizer | adamw8bit | adamw8bit | ✅ sama |
| noise / timestep | flowmatch / weighted | flowmatch / weighted | ✅ sama |
| EMA | none | none | ✅ sama |
| CFG / do_cfg | none | none | ✅ sama |
| caption_dropout | none | none | ✅ sama |
| fp8 / quantize | qfloat8 + te | qfloat8 + te | ✅ sama |
| trigger | Evelyn Reed | Evelyn Reed | ✅ sama |
| resolution | [512,768,1024] | [512,768,1024] | ✅ sama |
| **steps** | **168** | **config 2000 → window-cut ~750** (safetensors step750/epoch17) | 🔴 **BEDA — satu-satunya** |

## 🔴 LEVER YANG BIKIN 5FW2 MENANG = STEP COUNT (cuma itu)
Recipe **IDENTIK 100% kecuali steps**. Bukan luck / lever kecil acak — lever jelas:
- **5FW2: 168 step** = sweet-spot pendek → gak overfit → **0.0401**.
- **Boss: config 2000, ke-cut window ~750** (safetensors `step 750, epoch 17`) → **overtrain** person 14-img → **0.0414**.

🎯 **168 = 12 × 14 img** → **PERSIS formula `12×img` yang sama kayak Qwen person F2** (108 = 12×9).
Jadi 5FW2 pakai heuristik **12-step/img buat SEMUA ai-toolkit person task (Qwen + Z)**; boss biarin default gede → overtrain.

## 🔴 Implikasi buat pipeline kita
Template Z kita HEAD (`base_diffusion_zimage.yaml`) `steps: 2000` = **PERSIS config BOSS yang KALAH** (overtrain).
Kalau Z jalan via pipeline as-is → kita **niru BOSS (overtrain), BUKAN pemenang**.
→ Buat menangin Z: **steps = 12×img** (= 168 buat task ini) — **persis F2 step-fix, scoped ke Z person**.
Itu **satu-satunya** yang perlu diubah; network/conv/fp8/assistant_lora/dll template udah match winner (lihat KONFIRMASI_z_winner_5FW2.md).

## Ke-baca vs nggak (jujur)
- Semua ke-baca dari `config.yaml` (committed di repo boss) + `last_000000750.safetensors __metadata__`
  (software ai-toolkit 0.7.10, step750/epoch17, 0 dora, rank32 `[32,256]`, 240 modul). Gak ada yang ngarang.
- Boss gak punya `last.safetensors` (HTTP 404) — checkpoint-nya `last_000000250/500/750` → konfirmasi run ke-cut ~750 (bukan 2000 penuh).

## Jawaban akhir
- **Lever menang = step count (168 vs ~750).** Margin 3.1% tapi penyebab JELAS (overtrain), bukan luck.
- **Boss gak punya lever lebih bagus** — boss = recipe IDENTIK tapi overtrained. **5FW2 jelas unggul** (trained ke step yang bener).
- Niru 5FW2 (plain LoRA-32 / conv16 / **steps 12×img**) = bener; **gak ada lever boss yang perlu diambil**.
  Step-fix Z = lever kemenangan, **formula sama kayak Qwen person**.

### Sumber/bukti
- HF `...-956b14d3-...-5GU4Xkd3` : `checkpoints/config.yaml` (steps 2000) + `checkpoints/last_000000750.safetensors`
  (__metadata__ step 750, ai-toolkit 0.7.10, 0 dora_scale, 240 modul, rank 32). File: last_000000250/500/750 (no last.safetensors).
- Banding vs 5FW2 (KONFIRMASI_z_winner_5FW2.md: steps 168, sisanya identik) + `base_diffusion_zimage.yaml` HEAD (steps 2000).
