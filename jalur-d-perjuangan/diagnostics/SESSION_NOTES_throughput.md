# SESSION NOTES — Setup OneTrainer + Ukur Throughput (H100) — 2026-06-24

## CATATAN BUAT TRAINING ASLI (next session, JANGAN ubah sekarang)
- **Resolusi training asli WAJIB 1024** (match dataset native: T1 1024x1024, T2 1024x768 + boss).
  Committed config draft `resolution=512` itu PLACEHOLDER, bukan setting yg di-lock.
  "Jangan utak config" = cuma DoRA hyperparams (rank/alpha/stochastic_rounding/caption_dropout/decompose).
- Setting WAJIB yg BELUM ada di committed config (tambah sebelum train asli): stochastic_rounding=ON,
  caption_dropout=0.05 manual. (gak ngaruh throughput, ngaruh quality.)

## MEASUREMENT PLAN (sesi ini)
- Ukur step/menit DoRA buat Qwen & Z, di resolusi 1024 (PRIMARY/basis budget) + 512 (ratio/fallback).
- Buang warmup ~30s pertama (compile+caching+optim init). Qwen ukur 10mnt, Z 5-7mnt steady.
- Catat juga: durasi caching latent/text-encoder, durasi upload safetensors.
- Watch VRAM Qwen 20B @1024 -> kalau OOM lapor, jangan maksa.

## HASIL (FINAL — diukur 2026-06-24, torch 2.11.0+cu128 + shim, H100 80GB)
| model | res | download base | caching | throughput (step/mnt) | s/step | upload artifact | VRAM peak |
|-------|-----|---------------|---------|-----------------------|--------|-----------------|-----------|
| Qwen  |1024 | 40s (54GB)    | 48.1s   | **27.1** (cnt 271/600s; tqdm 28.6) | 2.22 | ~210MB | 27.8GB |
| Z     |1024 | 23s (31GB)    | 22.7s   | **44.9** (cnt 315/421s; tqdm 46.2) | 1.34 | ~72MB (gate) | 12.7GB |

(512 dibatalin atas instruksi Wen — dataset fix 1024, 512 buang waktu.)

### Konteks window (buat Wen itung step budget — TIDAK gue hitung, scope stop di sini)
- Qwen worst-case window 1.0h; Z worst-case 0.5h. Window = TOTAL (dl base + caching + train + upload).
- Validator cache base di /cache read-only → download base kemungkinan DI LUAR window (pending konfirm admin).
- DoRA params verified utuh: rank16/alpha16, decompose ON, AdamW, fp8 transformer, gc ON. Qwen 720 / Z 210 layer (match gate).
- CAVEAT: diukur cu128 (driver 535), bukan cu130. Workload bf16-bound (fp8 storage-only) -> delta cu128<->cu130 marginal,
  tapi kalau validator H100 driver >=580/cu130, throughput asli bisa beda tipis (~few %).
- Upload: artifact kecil (Qwen 210MB, Z 72MB) -> hitungan detik. BELUM diukur upload real (gak ada HF write token).
- Identifikasi dataset: T1=2ae5b33ce6138afe_train (44 img David), T2=515e7013c71fb202_train (36 img BrandEssence).

## ============ LANGKAH 2: TRAIN + EVAL DoRA (2026-06-24) ============
DoRA confirmed (720 dora_scale Qwen / 210 Z). Train eager (compile crash). Eval = container deterministic seed=42, single per ckpt.
Garis menang 3%: T1 <=0.08328 (boss 0.08585) | T2 <=0.03960 (boss 0.04083). Weighted=0.25*text+0.75*no_text.

| model | step | text_avg | no_text_avg | WEIGHTED | garis | vs boss | verdict |
|-------|------|----------|-------------|----------|-------|---------|---------|
| T1 Qwen | 1500 | 0.11945 | 0.10421 | **0.10802** | <=0.08328 | +25.8% (worse) | **KALAH** |
| T2 Z | 590  | 0.02883 | 0.03498 | **0.03344** | <=0.03960 | -18.1% (better) | **MENANG** |
| T2 Z | 860  | 0.02761 | 0.03282 | **0.03152** | <=0.03960 | -22.8% | **MENANG** |
| T2 Z | 1150 | 0.02672 | 0.03221 | **0.03083** | <=0.03960 | -24.5% | **MENANG** |

KESIMPULAN:
- T2 Z (logo): DoRA MENANG, nyebrang garis udah di 590 step (titik pertama), curve turun monoton 590->1150.
  Menang di <=860 step => REACHABLE eager worst-case window (Z 0.5h ~710 step budget). Compile fix TIDAK perlu utk Z.
- T1 Qwen (person): DoRA KALAH di 1500 (0.108 >> garis, +26% vs boss). Bukan isu step (1500>boss~1250).
  Catatan tuning (next session, BUKAN sekarang): LR=3e-4 baseline dipake; handoff flag DoRA LR-sensitif, saran 1e-4.

## ===== RETRY 2: Qwen plain-LoRA-128 5-lever (replikasi pemenang, minus CFG) — 2026-06-24 =====
Config: plain LoRA (decompose OFF) rank128/alpha128, 840 modul, adamw8bit LR1e-4 wd1e-5, EMA0.995,
multi-res[512,768,1024], LOGIT_NORMAL timestep, caption_dropout0.05, bs1, eager. Gap vs pemenang: CFG-train + timestep-mechanism.
| step | text_avg | no_text_avg | WEIGHTED | garis<=0.08328 |
|------|----------|-------------|----------|-----|
| 60  | 0.13538 | 0.10298 | 0.11108 | KALAH |
| 108 | 0.13684 | 0.10420 | 0.11236 | KALAH |
| 150 | 0.13674 | 0.10606 | 0.11373 | KALAH |
Banding: DoRA-1500 lama=0.10802 | boss T1=0.08585 | winner(task LAIN)=0.0716.
VERDICT (c): plain-LoRA-128 GAK membaik (0.111-0.114, malah sedikit > DoRA 0.108). Curve naik (makin buruk) sama step.
2 resep ekstrem beda (DoRA16/1500 vs LoRA128/108) konvergen ~0.11 -> differentiator BUKAN hyperparam LoRA.
Gap 0.11 vs boss 0.086 = FUNDAMENTAL (OneTrainer training mechanics vs ai-toolkit), bukan cuma CFG/timestep.

## ===== DIAGNOSTIK Qwen plateau + GATE EMA-off (2026-06-24) =====
Pipeline VALID: boss LoRA via pipeline kita = 0.0867 (≈ref 0.08585). base-only (zero-up) = 0.10934.
Semua LoRA OneTrainer kita ≈ base (DoRA 0.108, plain 0.112) -> awalnya nyaru "gak ke-apply".
Δweight test (comfy, actual merged delta, miss=0 di SEMUA = modul bener):
  DoRA-1500 Δ=5068 (OVERCOOK, LR3e-4x1500) | plain-s108 EMA-on Δ=59 (NULL, EMA0.995@108) | BOSS Δ=644.
Remap key (transformer.->diffusion_model.) = IDENTIK original -> key-naming BUKAN isu.
GATE EMA-off (plain-LoRA-128, LR1e-4, decompose OFF, multi-res, 264 step):
  | ckpt | Δweight | weighted |
  | s132 | 177 | (lihat di atas) |
  | s264 | 529 (~boss 644) | 0.11074 (≈base) |
VERDICT: EMA-off NAIKIN Δ ke boss-range (59->529) TAPI skor tetep ≈base. Adapter kebentuk ukuran+lokasi
benar tapi INCOHERENT. Pembeda menang Qwen = koherensi (CFG-train + timestep-mechanism) yg OneTrainer GAK punya.
-> OneTrainer mentok ~base buat Qwen walau adapter sehat. REKOMENDASI: ai-toolkit (tool pemenang, full-lever).
Z (DoRA-1150, 0.0308, MENANG) TIDAK disentuh sepanjang diagnostik.
