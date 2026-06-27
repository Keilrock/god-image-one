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

---
---

# FASE 4 — ASAL-USUL `flux.json default` (forensik git)

## Timeline commit `scripts/lrs/flux.json` (cuma 2 commit pernah sentuh)
| commit | tanggal | author | msg | default |
|--------|---------|--------|-----|---------|
| `c65c405` | 2026-05-29 | **besimray** | "Tournament winner repository - Commit: 97607210" (IMPORT boss) | `{}` (boss verbatim) |
| `e17c29e` | **2026-06-04 17:04 +07** | **Keilrock (Wen sendiri)** | "Update flux.json" | `{rank32/adamw/1000step/lr5e-5}` |

## 1. Kapan & di jalur mana diubah
- Diubah commit `e17c29e`, **4 Jun 2026**, **di `main`** SEBELUM jalur-b/c/d/e split.
- Konfirmasi: `e17c29e` = ancestor SEMUA branch (b ✅, c ✅, d ✅, e ✅). Makanya semua jalur kebawa default sama.
- Bukan dibuat di jalur tertentu — ke-inherit dari main ke semuanya.

## 2. Kenapa diubah (konteks)
- Bagian dari **batch 3 commit 4 Jun** (16:58–17:04) sama Keilrock:
  - `6db821a` person_config.json → recipe size-aware (xs/s/m/l/xl) **prodigy**, min_snr_gamma, canggih
  - `6ad862b` style_config.json → recipe size-aware **adamw** + noise_offset, canggih
  - `e17c29e` flux.json → default flat `{rank32/adamw/1000step/lr5e-5}`
- Person/style = recipe SDXL serius (size-aware, prodigy/adamw tuned). Flux = default **flat sederhana**, gak ada size-bucket, gak nyiru recipe juara flux.
- Commit msg "Update flux.json" **kosong rationale**. → **Intentional edit, TAPI isi generic/tebakan** (gaya SDXL-adamw ditempel ke flux), BUKAN diturunin dari recipe juara flux (yg baru kita tau = rank128/Lion/250 dari source boss).

## 3. Pernah di-test/dipakai turnamen? → **TIDAK**
- 11 Jun final round (branch b, peringkat 4) = **7 task, NOL flux**:
  T1 qwen(person), T2 z(logo), T3 sdxl(product), T4 sdxl(style), T5/T6 sdxl(logo), QF sdxl(person).
- Config flux default ini **gak pernah dieksekusi** di turnamen manapun yg kita punya datanya. Nyangkut sejak 4 Jun, idle.

## 4. Ada task Flux di 11 Jun? → **TIDAK ADA**
- Task Flux PERTAMA muncul di **18 Jun** (#4, a768272f) — turnamen yg lagi kita target.
- 18 Jun itu yg menang 5FW2 (miner eksternal) pakai **template boss** (rank128/Lion/250), BUKAN config kita.
- Repo kita (branch b) **gak ikut** ronde flux 18 Jun (b cuma ikut 11 Jun yg gak ada flux).

## KESIMPULAN FORENSIK
`flux.json default` (rank32/adamw/1000step/lr5e-5) = **edit sengaja tapi isi generic/tebakan** sama Wen di 4 Jun,
saat batch-tuning config SDXL person/style. **TIDAK PERNAH dipake/di-test** (gak ada task flux sampe 18 Jun, dan
repo kita gak ikut ronde itu). Praktis = **leftover yatim**, bukan recipe flux hasil eksperimen. Recipe juara flux
yg sebenernya (rank128/Lion/250) ada di TEMPLATE (`base_diffusion_flux.toml`) yg justru ke-override default ini.

→ Mendukung revert `default={}` = balik ke template juara, sbg baseline. (TAPI nunggu putusan Wen — belum di-edit.)

## STATUS: FASE 4 SELESAI — STOP. Belum edit apapun.

---
---

# FASE 5 — REVERT + ENV SETUP + THROUGHPUT GATE

## 1. Revert flux.json (commit 738f4ae)
`scripts/lrs/flux.json` default `{...}` → `{}`. Verified: base kita resolve ke template juara verbatim (rank128/Lion/250).

## 2. Env kohya Flux — pakai DOCKER (env persis boss)
- Boss prod pakai image `diagonalge/kohya_latest:latest` (BUKAN venv mentah). Pull OK.
- Env di image: **torch 2.1.2+cu121, diffusers 0.32.2, transformers 4.44.2, accelerate 0.33.0, lion-pytorch OK**. H100 80GB driver 535/cu12.2 → cu121 jalan.
- 🔑 Komponen flux **baked di image** `/app/flux/`: ae.safetensors (335MB), clip_l.safetensors (246MB), t5xxl_fp16.safetensors (9.8GB). Cuma perlu download unet.
- 🔴 **GOTCHA penting**: kode boss panggil `/app/sd-scripts/flux_train_network.py` (built-in image), TAPI dockerfile COPY vendored ke `/app/sd-script` (beda huruf 's'!). → **Boss sebenernya jalanin sd-scripts BAWAAN image, vendored repo UNUSED.** Vendored (fa6d8c08) internally inconsistent (`get_noise_pred_and_target() got multiple values for 'is_train'`) — GAGAL kalau dipaksa. Pakai built-in image = JALAN.

## 3. Base model
- `mhnakif/fluxunchained-dev` → file `fluxunchained-dev-fp16.safetensors` = DiT/unet (23.8GB, 780 tensor, format `double_blocks.*`). Download ke `/ephemeral/flux_models/unet.safetensors`, integritas MATCH.
- Dataset: train zip udah ada **caption .txt built-in** (11 png + 11 txt) → gak perlu llava auto-caption. Extract ke `/ephemeral/flux_run/img/1_lora style/` (repeats=1, folder kohya).

## 4. 🎯 THROUGHPUT GATE (config_gate.toml = template verbatim, max_train_steps=50)
Run: `docker ... cd /app/sd-scripts && accelerate launch flux_train_network.py --config_file config_gate.toml`
- Training JALAN ✅. num img 11, batch 4, grad_accum 2, 3 batch/epoch, 2 step/epoch.
- avr_loss start ~0.366-0.386 (sane).
- **Steady-state throughput: ~14.3 s/it** (per optimizer step; range 13.5–14.7). VRAM **58/80GB**, GPU util 100%.

### Estimasi full run
| | waktu |
|---|---|
| 250 step × 14.3s | ~59.6 min (train-loop) |
| + startup (load 24GB unet+10GB t5+cache latent/TE) | ~1 min |
| + 12 checkpoint save (fp32 ~3.5GB each) | ~3 min |
| **TOTAL full 250** | **~64 menit** |

### 🔴 Temuan kunci: throughput kita ≈ boss → window 1.0h CUT di ~epoch 120
- Full 250 step ≈ 64 min > **window turnamen 1.0h (60 min)**.
- → di turnamen, run ke-cut sekitar **step ~235-240 = epoch ~118-120** SEBELUM nyampe 250.
- **PERSIS cocok** sama checkpoint HF boss/5FW2 yg mentok `last-000120` (epoch 120). Konfirmasi: throughput H100 kita = throughput mereka, window naturally cap di epoch 120.

### Implikasi buat run kita
- Kita **gak kena window** (train lokal) → bisa full 250 atau lebih.
- TAPI checkpoint juara 5FW2 = epoch ~120 (yg ke-save terakhir di-window). Strategi: train full ~250, save tiap 10 epoch (kurva), eval checkpoint sekitar epoch 100-120 (zona pemenang) + yg lebih tinggi (karena kita bebas window, bisa cek apakah >120 lebih bagus).

## STATUS: FASE 5 GATE SELESAI — STOP. Throughput ~14.3s/it, full 250 ≈ 64 min. Nunggu go buat full train.
