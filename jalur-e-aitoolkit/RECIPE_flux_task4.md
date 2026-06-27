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

## STATUS FASE 1: recipe struktural identik (5FW2==BOSS). Hyperparameter di-strip dari header → lanjut Fase 2 (source code).

---
---

# FASE 2 — RECIPE FLUX LENGKAP DARI SOURCE CODE BOSS (bukan metadata!)

Source asli boss position-1 (5GU4) PUBLIK & gak di-strip:
`github.com/gradients-opensource/god-image-tourn-c43e1fc22c71be8f-20260618-position-1`

## 🔑 Cara recipe Flux dirakit (dari `scripts/image_trainer.py`)
1. Load template TOML: `scripts/core/config/base_diffusion_flux.toml` (semua default flux).
2. Override per-base-model dari `scripts/lrs/flux.json`, keyed `sha256(model_name)`.
3. **Base kita `mhnakif/fluxunchained-dev` → sha256 = `6445f395...` → override `{}` KOSONG.**
   → **TEMPLATE = RECIPE VERBATIM. Gak ada override LR sama sekali buat base kita.**
   (Override LR cuma kena base lain: 2 base dapet unet_lr 5e-5 / te 5e-6; satu dapet alpha 128 + 240 step. BUKAN base kita.)

## 🎯 RECIPE FLUX #4 FINAL (template verbatim — yang boss & 5FW2 jalanin)

### 1. Learning rate (train_t5xxl=True → 2 te_lr)
| | value |
|---|---|
| **unet_lr** | **0.00008** (8e-5) |
| **text_encoder_lr** | **[8e-6, 8e-6]** (CLIP-L, T5-XXL — dua-duanya 8e-6) |

### 2. Optimizer + scheduler
| | value |
|---|---|
| **optimizer_type** | **Lion** ⚠️ (BUKAN AdamW!) |
| optimizer_args | weight_decay=0.005, betas=(0.9,0.99) |
| lr_scheduler | cosine (num_cycles 1, power 1) |
| **warmup** | **TIDAK ADA** (lr_warmup_steps unset = 0) |

### 3. Timestep + flow + guidance (flux dev guidance-distilled)
| | value |
|---|---|
| timestep_sampling | **sigmoid** |
| discrete_flow_shift | **3.1582** |
| **guidance_scale** | **85.0** ⚠️ (SANGAT tinggi — normal flux train ~1.0; ini di-feed beneran ke model, confirmed `flux_train_network.py:342`) |
| model_prediction_type | raw |
| max_timestep | 1000 |
| loss_type | l2 (huber params ada tapi diabaikan krn l2) |
| noise_offset_type | Original |

### 4. Network args lengkap
| | value |
|---|---|
| network_module | networks.lora_flux |
| network_dim / network_alpha | **128 / 64** (rasio 0.5) |
| network_args | train_double_block_indices=**all**, train_single_block_indices=**all**, train_t5xxl=**True** |
| conv_dim / conv_alpha | **TIDAK ADA** (cuma SDXL pakai conv; flux pure linear LoRA) |
| dropout | **TIDAK ADA** di network_args (yg muncul `dropout:null` di metadata = default kosong) |
| apply_t5_attn_mask | **true** |
| t5xxl_max_token_length | 512 |

### 5. 🔴 Step / epoch (KOREKSI Fase 1!)
| | value |
|---|---|
| **max_train_steps** | **250** (cap yg di-set) |
| save_every_n_epochs | **10** (save by EPOCH, BUKAN step) |
| DIFFUSION_FLUX_REPEATS | **1** (constants.py) |
| train_batch_size / grad_accum | 4 / 2 → **eff batch 8** |
| steps/epoch | 11 img × 1 rep / 8 = **~2 optimizer step/epoch** |
| **`last-000120` artinya** | **EPOCH 120 ≈ ~240 optimizer step** (BUKAN 120 step! Fase 1 salah baca) |
| efektif vs cap | epoch 120 (~240 step) ≈ mentok max_train_steps 250 (~epoch 125). Window 1.0h cut tipis di ujung / nyaris kelar. |
| 12 checkpoint | epoch 10,20,…,120 (interval save 10 epoch) ✓ cocok sama yg di HF |

### 6. Versi kohya / sd-scripts
- Vendored di `scripts/sd-script/` (commit repo boss `fa6d8c08`, 2026-06-19). FLUX trainer = `flux_train_network.py` (punya `get_mod_vectors`, varian sd3/flux branch).
- requirements: **accelerate 0.33.0, transformers 4.44.0, diffusers 0.25.0, safetensors 0.4.4, bitsandbytes 0.44.0, lion-pytorch 0.0.6** (Lion!), sentencepiece 0.2.0, `-e .` (editable kohya_ss).
- ⚠️ Versi LAMA & beda total dari env ai-toolkit (torch 2.9/transformers 5.x). Flux butuh **environment terpisah** — pakai vendored sd-script langsung biar match.

### Setting lain yang penting (template)
- mixed_precision bf16, **full_bf16 true**, **save_precision float (fp32 → makanya tensor F32 3.47GB)**
- **NO fp8_base** (fp8 gak dipakai; t5xxl pakai t5xxl_fp16.safetensors)
- cache_latents + cache_latents_to_disk true, highvram true, gradient_checkpointing true, xformers true
- resolution 1024,1024 (single-res); bucket: no_upscale, reso_steps 64, min 256 max 2048
- **seed = 2** (template fixed — sama buat semua; jadi seed BUKAN pembeda 5FW2 vs boss)
- caption_dropout_rate 0.1 (ditambah di image_trainer.py, bukan template)
- caption_extension .txt, auto-caption mode "person" (flux selalu is_style=False di training_paths.py:46)
- ae/clip_l/t5xxl/unet di-split file (bukan single checkpoint)

## 🔑 IMPLIKASI: kenapa 5FW2 (0.0364) > boss (0.0372)?
**Recipe IDENTIK total** — base sama (override kosong), template sama, **seed sama (2)**. Dua-duanya jalanin
config yang sama persis. → Selisih 0.0364 vs 0.0372 (−1.9%) BUKAN dari recipe. Kemungkinan:
- **Beda hardware/precision non-determinism** (bf16 + xformers + Lion = non-deterministic walau seed sama), ATAU
- **beda titik cut window** (salah satu ke-cut di epoch beda, walau dua-duanya nyimpen sampe 120), ATAU
- **noise eval 2-img held-out**.
→ Praktis: **gak ada lever recipe yg bisa di-tweak dari data ini buat ngalahin boss** — recipe-nya udah identik.
Titik awal = template verbatim. Improvement harus dari eksperimen (LR/flow-shift/step/guidance), bukan jiplak.

## STATUS: FASE 2 SELESAI — recipe Flux LENGKAP ke-extract dari source. STOP, nunggu putusan strategi.

---
---

# FASE 3 — BANDING CONFIG FLUX: REPO KITA (jalur-e) vs BOSS

Cek apakah config Flux di repo kita udah identik boss, atau ke-modif. NO train, analisis doang.

## File config Flux di repo KITA (lokasi)
| file | isi | status vs boss |
|------|-----|----------------|
| `scripts/core/config/base_diffusion_flux.toml` | template recipe flux | ✅ **IDENTIK boss** (byte-for-byte) |
| `scripts/lrs/flux.json` | override LR per-base | 🔴 **BEDA di key `default`** (lihat bawah) |
| `scripts/image_trainer.py` (flux apply-block L107) | `for k,v in lrs_settings: config[k]=v` | ✅ IDENTIK (perubahan dethrone cuma di SDXL/aitoolkit, flux gak kesentuh) |
| `trainer/utils/training_paths.py` (flux→toml, L46) | routing flux + output/dataset path | ✅ IDENTIK (perubahan cuma Z/Qwen style-detect) |
| `dockerfiles/standalone-image-trainer.dockerfile` | env kohya flux | ✅ IDENTIK (cuma +lycoris buat SDXL DoRA, gak ngaruh flux) |

## 🔴 BEDA UTAMA: `flux.json` key `default`
```
KITA: "default": {"max_train_steps":1000,"network_dim":32,"network_alpha":32,"unet_lr":5e-5,
                  "text_encoder_lr":[5e-6,5e-6],"lr_scheduler":"cosine","optimizer_type":"adamw",
                  "optimizer_args":["weight_decay=0.01","betas=(0.9,0.99)","eps=1e-08"]}
BOSS: "default": {}
```
`data` (per-base hash) IDENTIK dua repo. Bedanya cuma `default`.

### 🔴🔴 EFEK ke base kita (`mhnakif/fluxunchained-dev`, hash `6445f395...`)
Routing: `merge_model_config(default, data[hash])` = `{**default, **data[hash]}`. Base kita `data[hash]={}` (kosong)
→ hasil merge = **`default` mentah**. Terus di create_config flux: `config[key]=value` buat tiap key → **OVERRIDE template**.

- BOSS: `default={}` → merge `{}` → template TIDAK di-override → **recipe juara verbatim** (rank128/Lion/250).
- KITA: `default={...}` → merge = default penuh → **template DI-HIJACK** → recipe beda jauh.

## Tabel diff recipe Flux EFEKTIF (yg bener-bener jalan)
| lever | BOSS = 5FW2 juara (0.0364) | KITA jalur-e (resolved) | beda? |
|-------|---------------------------|--------------------------|-------|
| network_module | networks.lora_flux | networks.lora_flux | sama |
| **network_dim (rank)** | **128** | **32** | 🔴 ¼ rank |
| **network_alpha** | **64** | **32** | 🔴 |
| network_args (train_t5xxl, all blocks) | train all double+single, train_t5xxl=True | SAMA (dari template, gak di-override) | sama |
| **optimizer_type** | **Lion** | **adamw** | 🔴 |
| **optimizer_args** | wd 0.005, betas(0.9,0.99) | wd **0.01**, betas(0.9,0.99), **eps 1e-08** | 🔴 |
| **unet_lr** | **8e-5** | **5e-5** | 🔴 |
| **text_encoder_lr** | **[8e-6,8e-6]** | **[5e-6,5e-6]** | 🔴 |
| lr_scheduler | cosine | cosine | sama |
| warmup | none | none | sama |
| timestep_sampling | sigmoid | sigmoid (template) | sama |
| discrete_flow_shift | 3.1582 | 3.1582 (template) | sama |
| guidance_scale | 85.0 | 85.0 (template) | sama |
| model_prediction_type | raw | raw (template) | sama |
| **max_train_steps** | **250** | **1000** | 🔴 4× |
| save_every_n_epochs | 10 | 10 (template) | sama |
| resolution / bf16 / full_bf16 / seed=2 | (template) | SAMA (template) | sama |
| dataset path / output path | get_image_training_images_dir / get_checkpoints_output_path | IDENTIK | sama |

## STATUS akhir: **(c→b) Infra/routing IDENTIK boss, TAPI recipe EFEKTIF BEDA JAUH**
Bukan "default boss belum disesuaikan" — malah KEBALIKAN: repo kita **udah dimodif** (`flux.json default`,
warisan dethrone jalur-b/d) ke recipe yg **BUKAN juara flux**. Kalau train flux #4 apa adanya SEKARANG →
dapet **rank32 / adamw / 1000-step / lr5e-5**, BUKAN recipe juara 5FW2 (rank128 / Lion / 250-step / lr8e-5).
Recipe efektif kita = generic SDXL-ish leftover, UNTESTED buat flux, jauh dari 0.0364.

## ⚠️ YANG PERLU DIBENERIN SEBELUM TRAIN
1. 🔴 **`scripts/lrs/flux.json` → set `"default": {}`** (samain boss). Ini bikin base kita resolve ke template
   verbatim = recipe juara 5FW2. SATU baris ubahan, fix paling bersih.
   - (Alternatif: hapus entry `6445f395` dari data — TAPI tetep butuh default={} biar fallback aman. Mending default={}.)
2. Path/routing/output/dockerfile: GAK ada yg perlu dibenerin (udah identik boss).
3. Setelah default={}: rank128/Lion/250/lr8e-5/guidance85/flow3.16 = baseline juara, baru eksperimen dari situ.

Catatan: rank32/adamw/1000-step yg sekarang BUKAN strategi sengaja buat flux (flux belum digarap per SESSION_STATE)
— ini murni leftover. Kecuali Wen sengaja mau coba recipe alternatif itu, harus di-revert ke template.

## STATUS: FASE 3 SELESAI — STOP. JANGAN edit flux.json / train. Nunggu putusan Wen.
