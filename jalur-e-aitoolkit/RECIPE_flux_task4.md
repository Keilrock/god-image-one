# FASE 1 — Task #4 FLUX social : EXTRACT RECIPE + BANDING 5FW2 vs BOSS

Extract header safetensors (metadata-only, range-request, GAK download 3.47GB). NO GPU, NO train.

## Fakta task #4
- task_id: `a768272f-fc01-457a-874d-06d6b15c5111` | model_type: **flux** | window **1.0h (KETAT)**
- base: `mhnakif/fluxunchained-dev` (FLUX.1-dev variant)
- dataset social_auraverse: train `39813ec781ce49a5` (11 img), test `3fbace30ed922645` (2 img)
- trainer: **KOHYA / sd-scripts** (`networks.lora_flux`), BUKAN ai-toolkit
- Skor 18 Jun: **5FW2Eaae 0.036437 (TERBAIK)** vs boss 5GU4Xkd3 **0.037169**. Target minggu ini: < 0.0372.

## Repo yang di-extract (checkpoint TERAKHIR/terbesar masing-masing)
- 5FW2 (menang): `...-a768272f-...-5FW2Eaae` → `checkpoints/last-000120.safetensors` (3.467GB)
- BOSS (banding): `...-a768272f-...-5GU4Xkd3` → `checkpoints/last-000120.safetensors` (3.467GB)
- Dua repo SAMA: checkpoint `last-000010` s/d `last-000120` (interval 10, max 120). **Dua-duanya mentok di 120** → kemungkinan ke-cut window 1.0h di titik yang sama.

---

## 🔴 TEMUAN UTAMA: METADATA DI-STRIP (cuma 16 keys, identik dua repo)

Header safetensors **diminimalkan** — kemungkinan kohya `--no_metadata` atau tournament sengaja strip
biar recipe gak bisa dijiplak. Yang SURVIVE cuma field arsitektur (modelspec.* + ss_network_*).

**Field hyperparameter yang DIMINTA tapi TIDAK ADA di header (gak bisa di-extract):**
`ss_learning_rate`, `ss_text_encoder_lr`, `ss_max_train_steps`/`ss_steps`, `ss_optimizer_type`,
`ss_lr_scheduler`+warmup, `ss_mixed_precision`/`ss_full_bf16`/`ss_fp8_base`, `ss_num_train_images`,
`ss_num_epochs`, `ss_timestep_sampling`, `ss_discrete_flow_shift`, `ss_model_prediction_type`,
`ss_guidance_scale`, `ss_seed`, `ss_max_grad_norm`, `ss_min_snr_gamma`.

→ Semua lever training (LR, step, optimizer, scheduler, flow-shift, guidance, seed) **HARUS dari
research komunitas / config kohya flux default**, BUKAN dari checkpoint ini.

---

## Tabel banding side-by-side (field yang SURVIVE)

| field | 5FW2 (menang 0.0364) | BOSS (0.0372) | beda? |
|-------|----------------------|---------------|-------|
| ss_network_module | networks.lora_flux | networks.lora_flux | sama |
| ss_network_dim (rank) | **128** | **128** | sama |
| ss_network_alpha | **64** (rasio 0.5) | **64** (rasio 0.5) | sama |
| ss_network_args | all double + all single, **train_t5xxl=True**, dropout null | all double + all single, **train_t5xxl=True**, dropout null | **IDENTIK** |
| modelspec.architecture | flux-1-dev/lora | flux-1-dev/lora | sama |
| ss_base_model_version | flux1 | flux1 | sama |
| modelspec.resolution | 1024x1024 (single-res) | 1024x1024 (single-res) | sama |
| modelspec.timestep_range | 0,1000 | 0,1000 | sama |
| ss_v2 | False | False | sama |
| jumlah tensor | 1632 (F32) | 1632 (F32) | sama |
| checkpoint max | last-000120 | last-000120 | sama |
| modelspec.date | 2026-06-19T00:23:58 | 2026-06-19T00:23:45 | beda **13 detik** (batch sama) |
| sshs_model_hash | 7d1b2e9e… | d0df0018… | beda (weight beda, wajar) |

### Modul yang dilatih (dari nama tensor, identik dua repo)
- `lora_unet_double_blocks_*`: **570 tensor** (semua double block) — dim 3072, rank 128
- `lora_unet_single_blocks_*`: **342 tensor** (semua single block) — dim 3072, rank 128
- `lora_te1_*` (CLIP-L): **216 tensor** — dim 768, rank 128
- `lora_te3_*` (T5-XXL): **504 tensor** — dim 4096, rank 128 → **train_t5xxl=True KONFIRMASI** (T5 beneran dilatih)
- TOTAL 1632 tensor, semua F32. **5FW2 == BOSS persis di struktur.**

---

## Hipotesis lever kunci

**Recipe yang RECOVERABLE = 100% IDENTIK antara pemenang (0.0364) dan boss (0.0372).**
Rank, alpha, block coverage, train_t5xxl, resolusi, timestep range, step count (120) — semua sama.

→ Selisih 0.0364 vs 0.0372 (**−1.9% relatif**) DIDORONG oleh lever yang **TIDAK ada di metadata**:
LR, T5 LR, timestep sampling, discrete_flow_shift, guidance_scale, optimizer, atau **seed/noise murni**.

**Dua kemungkinan:**
1. **Selisih = noise/seed.** Test set cuma 2 img → 1.9% gap bisa jadi variance held-out + seed beda.
   Kalau gini: recipe identik mana aja di range ini cukup, kita target replikasi struktur 5FW2 + LR/flow-shift default kohya-flux yang waras.
2. **Ada lever tersembunyi yang beda** (LR / flow-shift / guidance) yang gak ke-capture. Gak bisa
   dipastiin dari checkpoint — butuh research komunitas kohya-flux buat tau default & sweet-spot.

**Implikasi titik awal:** struktur recipe Flux udah PASTI (rank 128 / alpha 64 / train all blocks +
T5 + CLIP / 1024 single-res / ~120 step / window 1.0h). Yang BELUM pasti = LR, scheduler, timestep
sampling, flow-shift, guidance, optimizer → ini yang harus diisi dari research komunitas sebelum train.

## STATUS: FASE 1 SELESAI — STOP.
Recipe struktural ke-extract & dikonfirmasi identik (5FW2==BOSS). Hyperparameter training di-strip dari
header → next step = research komunitas kohya-flux buat ngisi LR/scheduler/flow-shift/guidance/optimizer,
baru putusin config titik awal. JANGAN setup kohya / train dulu.
