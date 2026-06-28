# GIT_HISTORY_MAP — Peta lengkap git history `Keilrock/god-image-one`

> Audit READ-ONLY git history (CPU, no GPU, no train). Dibuat untuk memetakan seluruh
> konteks: apa yang dikerjakan, kapan, di branch mana, recipe tiap task ke-lock di commit mana,
> dan mana yang sudah masuk pipeline default vs masih manual. Sumber: `git log`, `git show`,
> isi file di HEAD tiap branch. Tanggal pakai format repo (2026).

---

## 1. LINEAGE BRANCH

Lineage **LINIER**: tiap jalur dibangun di atas jalur sebelumnya. `jalur-e-aitoolkit` (HEAD) = kumulatif (memuat semua b+c+d+e).

```
c65c405  2026-05-29  IMPORT BOSS ("Tournament winner repository - 97607210")   ← akar / repo juara
   │
   ├─ 6db821a 06-04  Update person_config.json   ┐  (Wen, di main)
   ├─ 6ad862b 06-04  Update style_config.json    ├─ origin/main HEAD = e17c29e
   └─ e17c29e 06-04  Update flux.json (leftover) ┘
   │
   │   ├─(side, dead-end) calibration-pilot: 0b29fe0→a87992c→3caaf69 (06-10)
   │
   ├─ 9b39b80/f19a884 06-11  size-aware steps + style detection (Z/Qwen ai-toolkit)
   └─ 4fc5e1d 06-17  Fix step cap: Qwen 3000 / Z 2000      ← origin/jalur-b-sizeaware HEAD
        │
        ├─ 4a9952b 06-17  recipe tuning intel 11-Jun
        ├─ a5ddde9 06-18  routing per-kategori SDXL (HYBRID) + eval A/B
        ├─ 765e820 06-18  add lycoris_lora dep
        ├─ 98b5756 06-18  lycoris defensive (dockerfile + runtime fallback)
        ├─ 19ebe8a 06-18  harness FASE 0-2: env-override hooks (DETHRONE_SWEEP_*)
        ├─ 77b8041 / d454e49 06-18  runbook + docstring
        ├─ aff2ad0 06-19  OPSI A lock (recipe confirmed, no prod change)
        ├─ ac84a2c 06-19  person-SDXL cd0.10 via detektor person
        ├─ 4b2d628 06-19  restore SDXL style-gate champion (caption-keyword)
        └─ e185164 06-19  docs sync recap                 ← origin/jalur-c-dethrone HEAD
             │
             ├─ 73a8328 06-23  GATE 1: 1A PASS (OneTrainer+DoRA key-map ComfyUI, Qwen & Z)
             └─ a16886e 06-24  DAY2: throughput + DoRA + Qwen plateau diag  ← origin/jalur-d-perjuangan HEAD
                  │
                  ├─ 08f18a1 06-26  re-deploy ai-toolkit + EMA-off config
                  ├─ d02d437 06-26  Qwen person: EMA-off NYALIP pemenang (0.06825)
                  ├─ c7b5789/b45890e/85c96cc 06-26  task#3 art: EMA-off GAGAL (pakai boss EMA-on)
                  ├─ 291a954 06-26  SESSION_STATE
                  ├─ 3806802→7c229f9→7e0b3d1→ed31118 06-27  Flux #4 forensik recipe
                  ├─ 738f4ae 06-27  REVERT flux.json default ke {}   ← satu-satunya ubah PROD di jalur-e
                  ├─ 47a3091→83e3a4f→f2120e4 06-27  Flux #4 MENANG (0.03507)
                  ├─ bd4c197→ab4d2f2→784b9b5→8a6962d→39fdf3f→2381a06 06-27  SDXL logo audit (KALAH 0.0557)
                  └─ 834380c 06-27  HANDOFF + AUDIT_TODO + SESSION_STATE   ← origin/jalur-e-aitoolkit HEAD (SUBMISSION)
```

**Branch submission = `jalur-e-aitoolkit` (HEAD 834380c).** Kumulatif, memuat semua kerjaan.

---

## 2. PER-BRANCH: NGERJAIN APA + RECIPE YANG DI-LOCK

| Branch | Periode | Fokus | Yang dikerjain | Ubah file PRODUKSI? |
|---|---|---|---|---|
| **(main, c65c405)** | 05-29 | Import boss | Repo juara verbatim (commit 97607210). Semua template `base_diffusion_*` & lrs versi boss. | — (basis) |
| **main updates** | 06-04 | Wen tweak lrs | `person_config.json`, `style_config.json` di-update; `flux.json` default diisi leftover (rank32/adamw/1000) | lrs JSON (person/style/flux) |
| **jalur-b-sizeaware** | 06-11→17 | Size-aware ai-toolkit | `compute_aitoolkit_steps` (step proporsional jumlah gambar), style-detection Z/Qwen, step cap Qwen 3000 / Z 2000 | `image_trainer.py`, `training_paths.py` |
| **jalur-c-dethrone** | 06-17→19 | Recipe-lock SDXL + Qwen/Z | Routing per-kategori SDXL (`SDXL_NETWORK_BY_CATEGORY`), DoRA style/logo/design, plain-64 person/product, cd0.05 global + cd0.10 person, +caption_dropout 0.05 ke template Qwen/Z, rank 32→64, step floor 600→2000, lycoris dep + defensive, harness sweep env-gated (`DETHRONE_SWEEP_*`) | `image_trainer.py`, `training_paths.py`, `base_diffusion_qwen/zimage.yaml`, dockerfile, requirements |
| **jalur-d-perjuangan** | 06-23→24 | Z & Qwen via OneTrainer | GATE1 (DoRA key-map lolos ComfyUI), train DoRA OneTrainer, **Z menang 0.0308**, Qwen plateau (pivot ke ai-toolkit). Config OneTrainer = JSON manual terpisah. | **TIDAK ADA** (OneTrainer belum ke-integrate ke entrypoint/Dockerfile) |
| **jalur-e-aitoolkit** | 06-26→27 | Qwen EMA-off, Qwen-art, Flux, SDXL logo | Qwen person EMA-off menang (0.06825); Qwen-art EMA-off gagal→pakai boss EMA-on; Flux #4 menang (0.03507); SDXL logo audit (kalah 0.0557, ketauan config manual salah). | **HANYA `flux.json` (738f4ae revert ke {})** — sisanya NOTES/eksperimen manual |

🔴 **Catatan kunci:** setelah jalur-c, **pipeline produksi praktis BEKU**. `image_trainer.py` & semua template terakhir diubah di jalur-c (`4b2d628`, 06-19). jalur-d & jalur-e **tidak** mengubah pipeline kecuali `flux.json` revert. Artinya semua "kemenangan" jalur-d/e (Z OneTrainer, Qwen EMA-off) dihasilkan dari **config manual di luar pipeline**, dan **belum tentu kebawa** saat entrypoint dijalankan validator.

---

## 3. SEJARAH FILE RECIPE KUNCI (siapa ngubah, kapan, kenapa)

| File | Commit yang menyentuh | State di HEAD (jalur-e) |
|---|---|---|
| `scripts/lrs/flux.json` | c65c405 (boss: `default:{}`) → e17c29e (Wen +leftover default rank32/adamw/1000) → **738f4ae (revert ke `{}`)** | `default:{}`. Base dipakai `6445f395…`→`{}` → resolve ke template `base_diffusion_flux.toml` verbatim |
| `scripts/lrs/person_config.json` | c65c405 → 6db821a (Wen 06-04) | versi Wen 06-04. **Tidak disentuh dethrone**; network SDXL di-override runtime oleh routing |
| `scripts/lrs/style_config.json` | c65c405 → 6ad862b (Wen 06-04) | versi Wen 06-04. Sama, network di-override routing |
| `base_diffusion_flux.toml` | **hanya c65c405** | Boss verbatim = recipe juara Flux (rank128/Lion/guidance85/250step). Tidak pernah diubah |
| `base_diffusion_sdxl_person.toml` | **hanya c65c405** | prodigy `d_coef=1`, 25 epoch, min_snr 5, `networks.lora` dim/alpha=-1, cd=0 (cd & network di-override runtime) |
| `base_diffusion_sdxl_style.toml` | **hanya c65c405** | adamw, `networks.lora` dim -1, cd=0 (di-override routing→DoRA+conv+loraplus saat style) |
| `base_diffusion_qwen_image.yaml` | c65c405 (EMA on, linear128, 2500step) → 4a9952b (+`caption_dropout_rate:0.05`) | linear128, lr1e-4, adamw8bit, **`ema_config.use_ema: true` decay 0.995 (DARI BOSS, tak pernah diubah)**, cd0.05, steps2500 |
| `base_diffusion_zimage.yaml` | c65c405 → 4a9952b (+`caption_dropout_rate:0.05`) | linear32, lr1e-4, adamw8bit, cd0.05, steps2000 (ai-toolkit, BUKAN OneTrainer DoRA) |
| `scripts/image_trainer.py` | c65c405 → 9b39b80/f19a884 (size-aware) → 4fc5e1d (step cap) → 4a9952b (tuning) → a5ddde9 (routing SDXL) → 98b5756 (lycoris) → 19ebe8a (sweep hooks) → aff2ad0 → ac84a2c (person cd0.10) → **4b2d628 (terakhir, 06-19)** | Routing SDXL per-kategori, cd default 0.05 (L499), person cd0.10, size-aware ai-toolkit, backend routing (sdxl/flux→kohya, z/qwen→ai-toolkit). **Beku sejak jalur-c** |
| `trainer/utils/training_paths.py` | c65c405 → 9b39b80 → a5ddde9 → **4b2d628 (terakhir, 06-19)** | Pemilihan template SDXL via `detect_styles_in_prompts` (style→style.toml, else→person.toml); Z/Qwen via `_detect_style_from_captions`. Beku sejak jalur-c |

---

## 4. PER-TASK (6): RECIPE EFEKTIF SEKARANG (dari HEAD jalur-e) + PIPELINE vs MANUAL

> "Efektif via pipeline" = yang akan dihasilkan `create_config()` (image_trainer.py) saat entrypoint dijalankan validator. "Manual" = recipe eksperimen yang menang TAPI tidak tertanam di pipeline.

### Task 1 — Qwen person
- **Recipe pemenang (eksperimen):** ai-toolkit, winner 5FW2 verbatim **+ `ema_config.use_ema:false`** (EMA-OFF), linear128, 108 step, lr1e-4, adamw8bit, cfg6.0, skor **0.06825** (vs winner 0.07307).
- **Recipe via pipeline (template qwen HEAD):** **EMA-ON** (use_ema:true, decay0.995), linear128, steps size-aware (cap3000). EMA-off **tidak pernah di-commit ke template**.
- **Set di commit:** EMA-off cuma di config eksperimen manual (jalur-e, 08f18a1/d02d437) — bukan di file produksi.
- 🔴 **VERDICT: MANUAL / DEVIASI.** Pipeline menghasilkan EMA-ON (≈ baseline 0.08404 yang KALAH), bukan EMA-off pemenang. **Perlu commit `use_ema:false` ke `base_diffusion_qwen_image.yaml` kalau mau pemenang kebawa** (atau via mekanisme lain).

### Task 2 — Qwen-art (task3)
- **Keputusan akhir:** EMA-off GAGAL (0.07344 vs boss 0.07321) → **pakai boss EMA-ON verbatim** (0.07292). Base = **Qwen-Image-Jib-Mix**, 2500step (ke-cut window→1000), EMA-on, no trigger.
- **Recipe via pipeline:** template qwen HEAD = EMA-ON ✓ cocok keputusan. TAPI: (a) base ditentukan arg `--model` (Jib-Mix ok), (b) **steps di-override size-aware** (compute_aitoolkit_steps, bukan 2500 template), (c) ema decay 0.995 ✓.
- **VERDICT: PIPELINE ✓ (sebagian)** — arah EMA-on cocok template. ⚠️ Perlu cek FASE 1: apakah size-aware steps & cd0.05 mengubah hasil vs eksperimen. Eksperimen sendiri = ai-toolkit manual (belum diverifikasi via create_config).

### Task 3 — Z logo
- **Recipe pemenang (eksperimen):** **OneTrainer DoRA** rank16/alpha16 linear-only, AdamW, FP8, flowmatch, **0.0308** (vs boss 0.04083, −24%). Config = JSON manual OneTrainer.
- **Recipe via pipeline:** entrypoint route `z-image` → **ai-toolkit** (`/app/ai-toolkit/run.py`) + `base_diffusion_zimage.yaml` (**plain LoRA linear32, BUKAN DoRA**). **Tidak ada jalur OneTrainer di repo** (tak ada Dockerfile/routing OneTrainer).
- **Set di commit:** OneTrainer recipe di jalur-d (file JSON terpisah, tak masuk pipeline). Template ai-toolkit Z dari c65c405/4a9952b.
- 🔴 **VERDICT: MANUAL / TIDAK TER-INTEGRASI.** Pemenang Z (OneTrainer DoRA) **TIDAK reproducible via pipeline turnamen** — pipeline jalanin recipe ai-toolkit berbeda total. Guide resmi perkuat: Dockerfile Z/Qwen = ai-toolkit (`standalone-image-toolkit-trainer.dockerfile`), tak ada OneTrainer. **Gap integrasi besar.**

### Task 4 — Flux social
- **Recipe pemenang (eksperimen):** kohya, `networks.lora_flux` rank128/alpha64, **Lion** opt, guidance85, flow3.1582, lr8e-5/te8e-6, max_step250, train_t5xxl. Skor **0.03507** (vs boss 0.0372). Config = `config_full.toml` manual.
- **Recipe via pipeline:** flux.json `default:{}` (revert 738f4ae) + base `6445f395`→`{}` ⇒ resolve ke `base_diffusion_flux.toml` (boss verbatim = recipe juara). cd0.1 di-set runtime (L499).
- **Set di commit:** revert `738f4ae` (jalur-e) yang bikin pipeline resolve ke template juara.
- 🟢 **VERDICT: PIPELINE ✓ (paling sehat).** Pipeline resolve ke template = recipe juara. ⚠️ Verifikasi FASE 1: pastikan `base_diffusion_flux.toml` isi == recipe eksperimen (rank128/Lion/250/guidance85), dan cek apakah base lain (yang TIDAK di `data` flux.json) akan kena leftover — sekarang aman karena default `{}`.

### Task 5 — SDXL logo
- **Recipe eksperimen (manual, SALAH):** DoRA32+conv4 (H1) / plain (H2), **cd=0**, prodigy d_coef1.1 lr0.95, 45 epoch, min_snr6. Skor **0.0557** (KALAH vs boss 0.0465). cd=0 → overfit caption.
- **Recipe via pipeline (real):** template `base_diffusion_sdxl_person.toml` (prodigy **d_coef=1**, **25 epoch**, **min_snr5**) + routing kategori `logo` → **DoRA dim32 conv4/4 dora_wd dropout0** + **cd 0.05** (L499) + **LLaVA auto-caption**.
- **Boss:** plain `networks.lora` 32/32 no-conv + **cd 0.1**.
- **Set di commit:** routing logo-DoRA = a5ddde9; cd0.05 = a5ddde9/19ebe8a; person template = c65c405.
- 🔴 **VERDICT: EKSPERIMEN MANUAL = SALAH** (cd0 + skip LLaVA + network hardcoded). Pipeline asli ≠ eksperimen DAN ≠ boss (deviasi: cd0.05 vs 0.1, DoRA vs plain, epoch25/d_coef1 vs eksperimen 45/1.1). **Task ini perlu RE-VALIDATE via pipeline asli** (FASE setelah audit, pakai GPU 1 run).

### Task 6 — SDXL style / product (fortress)
- **Style (recipe lock jalur-c):** routing kategori `style` → DoRA dim32 conv4/4 dora_wd **loraplus_lr_ratio16** dropout0 + cd0.05. Template style.toml (adamw) di-override network oleh routing.
- **Product:** route `default` → plain LoRA-64, cd0.05 (fortress = seri boss).
- **Set di commit:** a5ddde9 (routing) + ac84a2c (person cd0.10, product tetap cd0.05).
- 🟢 **VERDICT: PIPELINE ✓.** Recipe tertanam di `image_trainer.py` routing (aktif saat training). Style = edge built-in jalur-c. Product = fortress.

---

## 5. FLAG: KETUKER / LEFTOVER YATIM / DEVIASI

| # | Jenis | Detail | Status |
|---|---|---|---|
| F1 | **Leftover yatim (sudah dibereskan)** | `flux.json` default rank32/adamw/1000 (Wen e17c29e, 06-04) — tak pernah dipakai/test. | ✅ REVERT 738f4ae → `{}` |
| F2 | 🔴 **Deviasi tak tertanam** | Qwen person EMA-off (pemenang 0.06825) **tidak di-commit** ke template; pipeline = EMA-ON (kalah). | ❌ BELUM. Perlu commit `use_ema:false` atau routing |
| F3 | 🔴 **Tidak ter-integrasi** | Z OneTrainer DoRA (pemenang 0.0308) **tak ada jalur di pipeline**; entrypoint Z → ai-toolkit plain-LoRA. | ❌ BELUM. Perlu integrasi OneTrainer ke Dockerfile+routing, ATAU terima recipe ai-toolkit Z |
| F4 | 🔴 **Eksperimen manual salah** | SDXL logo cd=0 manual (0.0557 overfit). Pipeline ≠ eksperimen. | ⚠️ Disimpan sbg NOTES (`sdxl_logo_NOTES_flawed_manual/`), bukan recipe aktif. Perlu re-validate pipeline |
| F5 | 🟡 **Deviasi pipeline vs boss (sengaja)** | SDXL logo pipeline (DoRA+cd0.05+prodigy d_coef1/ep25) ≠ boss (plain+cd0.1). Edge jalur-c −16.9% di T5 tapi cuma 1 task boss-valid; jalur-e gagal reproduce. | ⚠️ Recipe-wrong; butuh keputusan (replikasi boss vs pertahankan DoRA) |
| F6 | 🟡 **Manual belum diverifikasi** | Flux & Qwen-art eksperimen = config manual; belum di-diff vs `create_config`. | ⚠️ FASE 1 |
| F7 | 🟢 **lrs SDXL = versi Wen** | `person/style_config.json` (06-04) tak disentuh dethrone; network SDXL di-override routing runtime, jadi nilai network di JSON tak relevan untuk SDXL. | OK (by design) |

---

## 6. RINGKAS: MANA SUDAH KE-TIMPA KE PIPELINE DEFAULT, MANA MASIH MANUAL

| Task | Pemenang via pipeline `create_config`? | Keterangan |
|---|---|---|
| **Flux #4** | 🟢 YA (revert flux.json → template juara) | paling sehat; verifikasi isi template FASE 1 |
| **SDXL style** | 🟢 YA (routing jalur-c) | edge built-in |
| **SDXL product** | 🟢 YA (routing → plain-64) | fortress |
| **Qwen-art** | 🟡 SEBAGIAN (arah EMA-on cocok template) | base via arg; steps size-aware override; verifikasi FASE 1 |
| **SDXL logo** | 🟡 PIPELINE BEDA (DoRA+cd0.05, bukan eksperimen, bukan boss) | re-validate |
| **Qwen person** | 🔴 TIDAK (pipeline EMA-ON, pemenang EMA-off) | F2 — perlu commit |
| **Z logo** | 🔴 TIDAK (pipeline ai-toolkit, pemenang OneTrainer DoRA) | F3 — perlu integrasi |

**Inti:** 3 task aman via pipeline (Flux, style, product), 1 sebagian (Qwen-art), 1 beda-recipe (SDXL logo), **2 pemenang TIDAK kebawa pipeline (Qwen person EMA-off, Z OneTrainer)**. Pipeline produksi beku sejak jalur-c (06-19); semua kemenangan jalur-d/e dari config manual yang belum/tidak tertanam.

---

*Audit git-only, no GPU/train. Untuk verifikasi nyata config yang di-generate pipeline (diff vs recipe eksperimen), lihat FASE 1 (panggil `create_config()` di CPU).*
