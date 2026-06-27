# HANDOFF jalur-e — 2026-06-27 (sebelum matiin VM)

Branch: `jalur-e-aitoolkit`. Rangkuman LENGKAP kerjaan hari ini. Baca ini + SESSION_STATE.md dulu next session.

═══════════════════════════════════════════════
## SKOR TURNAMEN jalur-e
═══════════════════════════════════════════════
| Task | Status | Angka |
|------|--------|-------|
| #2 Qwen person | ✅ MENANG | EMA-off 0.06825 vs winner 0.07307 (−6.6%) |
| #3 Qwen-art | ✅ setara boss | pakai boss EMA-on 0.07292 |
| Z logo (jalur-d) | ✅ MENANG | DoRA 0.0308 vs boss 0.04083 (−24%) |
| **#4 Flux social** | ✅ **MENANG** | epoch120 **0.03507** vs juara 0.0364 (−3.8%), boss 0.0372 (−5.6%) |
| **SDXL logo** | 🔴 **BELUM kelar** | best kita 0.0557 vs boss 0.0465 (KALAH ~20%) — diagnosa selesai, fix belum |

═══════════════════════════════════════════════
## FLUX #4 — ✅ BERES MENANG
═══════════════════════════════════════════════
**Forensik flux.json**: default `{rank32/adamw/1000step/lr5e-5}` = **leftover yatim** commit e17c29e (Keilrock, 4 Jun, di main), batch update SDXL config — GAK PERNAH dipake/test (0 task flux di 11 Jun, repo gak ikut 18 Jun). **Di-REVERT ke `{}`** (commit 738f4ae) → base `mhnakif/fluxunchained-dev` resolve ke template juara verbatim.

**Recipe juara** (`base_diffusion_flux.toml` verbatim): rank128/alpha64, **Lion** (wd0.005), unet_lr 8e-5/te_lr 8e-6, **guidance 85**, flow-shift 3.1582, sigmoid, train_t5xxl, 1024, seed 2, 250 step.

**Env**: docker `diagonalge/kohya_latest` (torch2.1.2/diffusers0.32). 🔴 GOTCHA: pakai sd-scripts **BUILT-IN image** (`/app/sd-scripts/`), BUKAN vendored repo (`/app/sd-script`, beda 's', internally broken). Train via `flux_train_network.py`.

**Hasil**: train 250 step (~62 min, ~14.3 s/it). Eval validator asli (`gradientsio/image-evaluator:basilica` + redisfix). Sanity reproduce PERSIS (5FW2 0.03636 vs off 0.036437; boss 0.03717 vs 0.037169).
| ckpt | weighted | vs juara 0.0364 |
|------|----------|------|
| epoch 110 | 0.03506 | −3.8% |
| **epoch 120 (SUBMISSION)** | **0.03507** | **−3.8%** |
| epoch 125 (bonus) | 0.03506 | −3.8% |
- Kurva flat ~0.0350 mulai epoch 110. >120 gak nambah. **Submission = `last-000120.safetensors`**.
- Menang walau recipe identik = non-determinism run-to-run (bf16+Lion+xformers). 1 run, variance band ~0.035-0.037, margin ke boss lebar.

**HF backup**: `keilrockstars/flux-task4-jalur-e` (private) — ckpt 110/120/125 + config_full.toml + RESULTS + 8 eval json. ✅
**Detail**: `RECIPE_flux_task4.md` (forensik fase1-5) + `RESULTS_flux_task4.md`.

═══════════════════════════════════════════════
## SDXL LOGO — 🔴 BELUM KELAR (diagnosa done, fix pending)
═══════════════════════════════════════════════
Target: logo #6 18-Jun (anima-pencil), boss MENANG 0.04653. Logo = satu-satunya task masih KALAH.

**Fase A (banding)**: HF metadata boss #6 (stripped) = `networks.lora` plain 32/32, conv=0 (person route, mapping 228). Recipe kita (jalur-e, asal jalur-c) = lycoris **DoRA** 32/32 + conv4, no loraplus. Beda = NETWORK. Detail: `RECIPE_sdxl_logo.md`.

**Fase B1 (sanity)**: env kohya share flux. base anima-pencil (diffusers 6.5GB) + dataset (train 7489f54c 18img, test eb52ad46 3img). **Sanity boss #6 = 0.04652 vs official 0.04653 PERSIS** → pipeline SDXL VALID. trigger word #6 = `Brandmark_Essentials`.

**Fase B2 (H1 vs H2) — 🔴 PAKAI CONFIG MANUAL = SALAH**:
- H1 (DoRA+conv) vs H2 (plain) — keduanya OVERFIT, best epoch25 ~0.0557, last 0.059-0.065. KALAH boss 0.0465.
- H1≈H2 → network BUKAN lever.
- ⚠️ Config dibikin MANUAL (bukan pipeline asli) → caption_dropout=0 (kelewat).

**Fase B3 (diagnostik trajektori boss)**: eval ckpt boss #6 per epoch:
| epoch | boss | kita H2 |
|-------|------|---------|
| 10 | **0.0456** | 0.0613 |
| 20 | 0.0465 | 0.0558 |
| 40 | 0.0553 | 0.0604 |
| last(submit) | **0.0465** | 0.0652 |
- Boss optimal **epoch 10 (0.0456)**, boss "last" = best-early-pick (~epoch 10-20), boss JUGA overfit abis itu.
- Same-epoch10: boss 0.0456 vs kita 0.0613 → **VERDICT: RECIPE-WRONG** (bukan window-cut, bukan ckpt-select).

**Fase B4+B5 (AKAR + audit pipeline)**: 🔴 **TEMUAN KRITIS**:
- Eksperimen pakai config MANUAL, BUKAN pipeline asli → kelewat runtime override.
- **caption_dropout_rate**: manual gue **0** ❌ | pipeline asli kita **0.05** (image_trainer.py:499) | **boss 0.1** (image_trainer.py:307, override di akhir).
- **LLaVA auto-caption** (auto_caption_dataset): pipeline augment .txt + llava caption (`original + ", " + llava`, butuh /opt/models/llava 13GB). Manual pakai .txt mentah.
- **network routing**: pipeline auto-detect category="logo"→DoRA. Manual hardcode.
- Confirmed via `create_config()` asli: pipeline jalur-e logo = DoRA + cd0.05 + prodigy d_coef1.1/lr0.95/45ep/batch8/min_snr6/seed2951032222.
- 🔴 **Pipeline asli KITA ≠ BOSS** (2 deviasi dethrone): cd **0.05 vs 0.1**, network **DoRA vs plain**.

**Artefak logo (di /ephemeral, BELUM fix — JANGAN dipake sbg config aktif)**:
- /ephemeral/sdxl_logo/out_h1, out_h2 (9 ckpt each, config MANUAL cd=0 — flawed)
- /ephemeral/boss6_ckpts (boss #6 ckpt referensi)
- config_h1/h2_full.toml = MANUAL (cd=0), config_h1/h2_gate.toml. **Ini NOTES doang, bukan recipe final.**
- HF backup: lihat AUDIT_TODO. (Logo ckpt = referensi eksperimen flawed.)

═══════════════════════════════════════════════
## 🔴 TEMUAN KRITIS LINTAS-TASK (baca!)
═══════════════════════════════════════════════
**Eksperimen logo B2 pakai config MANUAL, bukan pipeline asli repo.** Turnamen jalanin pipeline asli
(`image_trainer.py` → prepare_dataset + auto_caption LLaVA + create_config runtime override + train).
Config manual KELEWAT: caption_dropout (line 499/307), LLaVA caption augmentation, network auto-routing.
→ **SEMUA eksperimen ke depan HARUS via pipeline asli** (`create_config()` / entry point), JANGAN rakit toml manual.
→ Flux #4 JUGA pakai config manual (config_full.toml) — TAPI flux template gak ada runtime cd-override
   yg signifikan + caption built-in (gak ada llava re-caption di flux? CEK). Hasil flux menang & sanity persis,
   jadi kemungkinan OK — tapi MASUK AUDIT (lihat AUDIT_TODO).

═══════════════════════════════════════════════
## ENV / INFRA (di /ephemeral — KE-WIPE pas VM mati)
═══════════════════════════════════════════════
- Docker images (ke-wipe, re-pull/build next session):
  - `diagonalge/kohya_latest` (kohya train flux+sdxl, flux components baked di /app/flux/)
  - `gradientsio/image-evaluator:basilica` (eval) → BUTUH redisfix: `pip install redis asyncpg minio boto3 tenacity aiohttp httpx loguru python-dotenv` → tag `image-evaluator:basilica-redisfix`
  - `sdxl-trainer-deps` (kohya + trainer pip deps, NO llava) — buat jalanin create_config/pipeline asli
  - trainer image PENUH (pipeline asli + llava 13GB) BELUM dibuild — perlu buat re-run logo via pipeline
- Eval runner: `run_eval_flux.sh`, `run_eval_sdxl.sh` (di /ephemeral, ke-wipe — logika di RESULTS docs).
- HF token + GitHub PAT: minta Wen (jangan commit).
- ⚠️ hf_xet uninstall (crash). /cache symlink ke /ephemeral.

═══════════════════════════════════════════════
## NEXT SESSION (prioritas)
═══════════════════════════════════════════════
1. **SDXL logo fix** (front terakhir 6/6): re-run VIA PIPELINE ASLI (DoRA+cd0.05+llava). Kalau kalah → A/B
   ke recipe boss (plain+cd0.1) lewat env `DETHRONE_SWEEP_CD` / edit image_trainer.py. Lihat AUDIT_TODO.
2. **AUDIT** semua task (Qwen/Flux): manual vs pipeline asli? (lihat AUDIT_TODO.md). NO GPU.
3. Re-setup env (docker re-pull/build + redisfix).
