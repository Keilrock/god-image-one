# FASE 3 — EMA-OFF RESULT (re-deploy 2026-06-26)

## 🎯 VERDICT: EMA-off NYALIP PEMENANG (gerbang ~0.073 LEWAT)

Recipe winner 5FW2Eaae VERBATIM, satu-satunya beda `ema_config.use_ema: false`.
Pipeline eval = pipeline yang tervalidasi Fase 2 (winner reproduce 0.07307 PERSIS).
Test set 106ffec9 (img 7 hold-out), master_seed 42, weighted = 0.25*text + 0.75*no_text.

| ckpt | text | no_text | WEIGHTED | Δweight | vs winner 0.07307 | vs EMA-on baseline |
|------|------|---------|----------|---------|-------------------|--------------------|
| EMA-off c36  | 0.06275 | 0.07009 | **0.06825** | 198.35 | **−6.6% (MENANG)** | EMA-on c36 0.08404 → −18.8% |
| EMA-off c72  | 0.06735 | 0.07366 | 0.07209 | 253.66 | −1.3% (menang) | EMA-on c72 0.08474 → −14.9% |
| EMA-off c108 | 0.06359 | 0.07319 | **0.07079** | 272.95 | **−3.1% (MENANG)** | EMA-on c108 0.08446 → −16.2% |
| WINNER (pemenang, Fase 2 reproduce) | 0.05375 | 0.07950 | 0.07307 | 247.10 | — | — |
| boss | — | — | 0.0867 | — | — | — |

## Kesimpulan
1. **MAGNITUDE**: EMA-off Δweight (c108=272.95) ≈ pemenang (247.10), confirm hipotesis handoff
   (EMA-on cuma ~82, 3× kekecilan — EMA 0.995 di HEAD ngecrush magnitude saved weight).
2. **ARAH/KOHERENSI**: loss EMA-off 0.068-0.072 NYAMAIN-DAN-LEWATIN pemenang 0.07307.
   Beda dari pelajaran OneTrainer (magnitude cocok tapi arah salah → tetep ~base):
   di sini magnitude DAN arah dua-duanya bener. ai-toolkit + CFG-train + timestep weighted + EMA-off = koherensi.
3. **c36 = TERBAIK (0.06825)** — gak monoton; early checkpoint malah terbaik (9-img, less overfit).
4. Gerbang handoff "loss ~0.073 = buffer aman 🎯" → TERCAPAI & terlampaui. Buffer vs boss 0.0867 = −21%.

## Δweight metric
Frobenius norm dari merged delta `scale·(B·A)` per modul (840 modul), scale=alpha/rank=128/128=1.0.
Reproduksi winner handoff persis (gue 247.10 vs handoff 246.4) → metrik valid.

## Artefak (di /ephemeral — BELUM ke-push HF, butuh HF token kalau mau diselametin dari wipe)
- /ephemeral/output/ema_off/ema_off_000000036.safetensors (c36, TERBAIK) 2.36GB
- /ephemeral/output/ema_off/ema_off_000000072.safetensors (c72) 2.36GB
- /ephemeral/output/ema_off/ema_off.safetensors (c108) 2.36GB
