# HANDOFF — SN56 jalur-d — Status per 24 Jun 2026 (DAY 2, lanjutan handoff H100)

> Baca ini DULU. Lanjutan dari HANDOFF_jalur-d_H100. Sesi ini = throughput + train DoRA + eval + diagnostik
> mendalam kenapa Qwen plateau. Verdict final: **pivot ai-toolkit buat Qwen**. Z UDAH MENANG (jangan utak).

---

## SIAPA & APA
- User = **Wen** (keilrockstars, hotkey 5FW81h8N). Casual Jakarta Indo (gw/lo).
- Project: SN56 Bittensor G.O.D image tournament. Dethrone boss (hotkey 5GU4Xkd3).
- Repo: github.com/Keilrock/god-image-one. Branch kerja: **jalur-d-perjuangan**.

## POSISI SEKARANG (1 paragraf)
**Z (logo, Z-Image): MENANG** — DoRA-1150 via OneTrainer = **0.0308** vs garis 0.0396 (boss 0.04083, −24%). Solid, terverifikasi multi-checkpoint, JANGAN diutak.
**Qwen (person): BUNTU di OneTrainer.** Semua percobaan (DoRA-1500, plain-LoRA-128 5-lever, plain-LoRA EMA-off)
mentok ≈ base (0.108-0.115) vs boss 0.0867. Diagnostik DEFINITIF (lihat §DIAGNOSTIK): pipeline valid, adapter
ke-apply ke modul bener dgn magnitude bener (Δ 529 ≈ boss 644) — TAPI **incoherent**. Gap = koherensi yang
OneTrainer gak bisa kasih (**CFG-training + mekanisme timestep-weighting**). **Pivot: ai-toolkit buat Qwen** (besok).

## DIAGNOSTIK Qwen — RANTAI ELIMINASI (semua TERBUKTI, angka real)
Referensi: boss LoRA via pipeline kita = **0.0867** (≈ ref 0.08585 ✓ pipeline valid). base-only (zero-up boss) = **0.10934**.

| # | hipotesis | test | hasil | status |
|---|---|---|---|---|
| 1 | pipeline eval rusak | eval boss LoRA di test-set kita | 0.0867 ≈ 0.08585 | ❌ RULED OUT (pipeline bener) |
| 2 | key-naming salah (transformer. vs diffusion_model.) | remap → format ai-toolkit, re-eval | IDENTIK original (0.108/0.112) | ❌ RULED OUT (comfy handle dua format) |
| 3 | modul nyasar (struktural-b) | Δweight test: miss count | **miss=0** semua, modul = persis boss | ❌ RULED OUT (apply ke modul bener) |
| 4 | EMA-null | Δweight EMA-on vs EMA-off | 59 → 529 (boss-range) | ✅ TRUE tapi gak cukup |
| 5 | **koherensi gap (CFG+timestep)** | gate EMA-off: Δ boss-range tapi skor? | **Δ529 / skor 0.111 ≈ base** | ✅ **INI BIANGNYA** |

**Δweight (delta aktual yg comfy merge ke bobot, via container comfy):**
| model | Δweight | eval weighted | catatan |
|---|---|---|---|
| DoRA-1500 (LR3e-4) | **5068** | 0.10802 | OVERCOOK (8× boss) → noise |
| plain-LoRA-128 EMA-on s108 | **59** | 0.11236 | NULL (EMA0.995@108) |
| plain-LoRA-128 EMA-off s132 | 177 | 0.11504 | sehat, incoherent |
| plain-LoRA-128 EMA-off s264 | **529** | 0.11074 | ~boss magnitude, TETEP incoherent |
| **BOSS** | **644** | **0.0867** | coherent → menang |
| base-only | — | 0.10934 | garis nol |

**Kesimpulan:** boss Δ644→0.0867 ("ngerti orang"); kita Δ529→0.111 (ukuran+lokasi sama, ARAH salah). Magnitude/
EMA/apply BUKAN pembeda — KOHERENSI yang beda. OneTrainer gak punya CFG-train + timestep-weighting (bell-curve
loss-weight; ai-toolkit "weighted" = uniform-sample + loss-weight tengah, gak ada padanan persis di OneTrainer).

## RECIPE PEMENANG 5FW2Eaae (kebongkar, 3 kategori, tournament 18 Jun) — di winner-recipes/
| aspek | Qwen person (0.0716) | Z person (0.0401) | SDXL style (0.0455) |
|---|---|---|---|
| metode | **plain LoRA** (BUKAN DoRA) | plain LoRA | plain LoRA + conv4 + LoRA+ |
| trainer | **ai-toolkit 0.7.10** | ai-toolkit 0.7.10 | **kohya** |
| rank/alpha | 128/128 | 32/32 | 32/32 |
| steps | **108** | 168 | (kohya meta stripped) |
| LR/opt | 1e-4 / adamw8bit | 1e-4 / adamw8bit | LoRA+ ratio16 |
| EMA | **0.995 ON** | ❌ | ? |
| CFG-train | **cfg 6.0 ON** | ❌ | ? |
| multi-res | [512,768,1024] | [512,768,1024] | (kohya bucket) |
| timestep | weighted | weighted | ? |
**Insight:** trainer per-ARSITEKTUR (DiT→ai-toolkit, SDXL→kohya). NOL DoRA di semua. EMA+CFG cuma Qwen.

## PLAN BESOK (fase 0-2)
- **Fase 0 — RE-CEK Z (penting):** Z juga DiT+OneTrainer. Kemenangan 0.0308 HARUS diverifikasi asli, bukan
  artefak base. Test: eval base-only Z (zero-up) vs Z-LoRA kita vs boss-Z + Δweight test Z. Kalau Z-LoRA Δ
  sehat & skor < base & < boss → menang asli (kemungkinan besar IYA, beda dari Qwen — tapi WAJIB konfirm
  sebelum andelin Z). Kalau Z ternyata artefak juga → Z belum menang, re-evaluasi.
- **Fase 1 — setup ai-toolkit:** clone ostris/ai-toolkit, venv (HATI cu128/driver-535, mungkin perlu workaround
  kayak OneTrainer; lihat §INFRA), convert dataset → format ai-toolkit (folder img+txt, multi-res).
- **Fase 2 — replikasi recipe Qwen pemenang:** plain-LoRA-128, 108 step, EMA0.995, CFG6.0, multi-res, adamw8bit
  LR1e-4, timestep weighted, trigger "David". Eval vs garis 0.08328 (boss 0.08585). INI jawaban inti DoRA-vs-LoRA.

## 🔴 DEADLINE — KAMIS 25 Jun 15:00 UTC: SUBMIT jalur-c (IRREVERSIBLE)
Paralel sama eksperimen. Checklist: repo UDAH publik ✓ · re-cek IMAGE_STYLES masih 72 · update endpoint
commit-hash → **e185164** (pas submit) · jalur-b-sizeaware tetep utuh. JANGAN utak jalur-c demi jalur-d.

## INFRA (H100 sesi ini — buat reproduce)
- GPU H100 80GB, driver **535/CUDA12.2** (BUKAN ≥580). Root 100GB / **ephemeral 750GB** → SEMUA ke ephemeral.
- **torch 2.11.0+cu128 + shim** (cu130 mustahil driver 535). Shim: adam/adamw_extensions.py
  `_cuda_graph_capture_health_check`→`_accelerator_graph_capture_health_check`. (lihat onetrainer_deps.md)
- **eager (compile=OFF)** — compile crash inductor di graph fp8+caption_dropout. Eager ~1.9× lambat, hasil sama.
- bitsandbytes perlu di-pip (ADAMW_8BIT). block_wise=True WAJIB (default False → crash bf16 grad).
- Docker data-root + HF cache + venv + model + output SEMUA ke /ephemeral (root cuma 100GB).
- Eval: container gradientsio/image-evaluator:basilica, deterministic master_seed=42 (internal 10-seed avg),
  MODELS=HF repo (LoRA di checkpoint/last.safetensors), base mount ke diffusion_models (skip download lambat).

## ARTIFACT (HF: keilrockstars/god-image-dora-gate, privat)
- Z DoRA-1150 (MENANG 0.0308), + 3 winner model 5FW2Eaae (qwen/z/style) buat referensi.
- HF repo eval (privat): god-image-dora-T1-qwen, -T2-z-s590/860/1150, diag-* (sementara).

## CARA KERJA
Wen pegang VM + Claude Code (eksekusi). Chat = strategi. Z JANGAN re-train. Selalu kasih konteks infra
(branch rules, ephemeral, cu128+shim) ke Claude Code. STOP+lapor tiap blocker, jangan stuck diem.
