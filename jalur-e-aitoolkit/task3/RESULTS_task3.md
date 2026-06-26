# FASE C — Task #3 Qwen-Jib-Mix art/style : EMA-off TRAIN + EVAL (2026-06-26)

## 🔴 VERDICT: EMA-off GAK menang buat art/style (kalah tipis). Boss EMA-on tetep pilihan.

Pipeline VALID (sanity boss reproduce 0.07321 vs official 0.07292, +0.4%) → angka dipercaya.
EMA-off terbaik (c1000 = 0.07344) > boss reproduce (0.07321) > boss official (0.07292).
Beda ~0.0002 (dalam noise 2-img held-out), TAPI di sisi yang salah. Temuan EMA-off person GAK transfer ke art.

## Setup
- base: Qwen-Image-Jib-Mix | test 93e6a3fa (2 img) | NO trigger | master_seed 42, weighted 0.25*text+0.75*no_text
- recipe EMA-off = boss-art 5GU4 VERBATIM, satu beda use_ema:false. Target 1100 step.
- 🔴 Boss config tulis 2500 TAPI checkpoint mentok last_000001000 → boss ke-cut window 1.25j @1000.
  Winner = step 1000 EMA-on. Kita samain target ~1100 (1000 + buffer).

## Tabel hasil

| LoRA | text | no_text | WEIGHTED | Δweight | vs boss official 0.07292 |
|------|------|---------|----------|---------|--------------------------|
| **BOSS 5GU4 c1000 (EMA-on, sanity)** | 0.05958 | 0.07775 | **0.07321** | 487.72 | +0.4% (reproduce OK) |
| EMA-off c600 | 0.08568 | 0.07054 | 0.07433 | 426.98 | +1.9% |
| EMA-off c800 | 0.08299 | 0.07172 | 0.07454 | 471.90 | +2.2% |
| **EMA-off c1000 (terbaik)** | 0.07969 | 0.07136 | **0.07344** | 530.63 | +0.7% (kalah tipis) |
| EMA-off c1100 | 0.08004 | 0.07187 | 0.07391 | 572.07 | +1.4% |

## Analisis
1. **SANITY PASS** — boss reproduce 0.07321 ≈ official 0.07292 (+0.4%, noise 2-img). Pipeline Jib-Mix VALID.
2. **EMA-off gak menang art** — best 0.07344 vs boss 0.07321 (pipeline sama). Seri/mepet, EMA-off sisi salah.
3. **Magnitude naik, loss nggak** — Δweight EMA-off c1000 (530.63) > boss EMA-on (487.72), tapi loss TETEP kalah.
   Echo pelajaran OneTrainer: magnitude lebih gede ≠ loss lebih bagus. Buat art, EMA-off cuma nambah magnitude tanpa koherensi ekstra.
4. **Breakdown text vs no_text** — EMA-off text-guided JAUH lebih buruk (0.080 vs boss 0.0596) tapi no_text sedikit lebih baik (0.0714 vs 0.0778). Karena weighted 0.75 ke no_text, net ≈ seri. Boss EMA-on ikutin prompt lebih bagus (text loss rendah).
5. **Kurva step (EMA-off)** — flat 0.0734-0.0745. c1000 sweet spot, c1100 naik dikit (mulai overfit). 600→1000 turun tipis, gak ada gain signifikan dari step lebih banyak. → buat submission step-scaling art: ~1000 cukup, lebih dari itu sia-sia.

## Kenapa beda dari person (#2)?
- Person: EMA-off menang telak (0.06825 vs winner 0.07307). EMA-on crush magnitude di step rendah (108) → EMA-off benerin magnitude+koherensi.
- Art: di 1000 step, EMA averaging udah "matang" (10× step person), magnitude EMA-on udah sehat (487). EMA-off cuma over-shoot magnitude (530) tanpa nambah koherensi → malah text-following turun.
- → EMA-off = lever PERSON-specific (low-step identity). Art/style (high-step) gak butuh.

## REKOMENDASI (nunggu putusan Wen)
Per gerbang prompt: EMA-off > 0.07292 → **fix pakai EMA-ON boss verbatim** (proven 0.07292), TANPA tes ulang. Hemat.
Recipe boss-art 5GU4 apa adanya (EMA on, 1000 step, no trigger, Jib-Mix) = submission #3.
EMA-off gak perlu di-pursue lebih lanjut buat art.

## Artefak (di /ephemeral, BELUM push HF)
- /ephemeral/output_t3/ema_off_t3/ema_off_t3_0000006/8/10*.safetensors + ema_off_t3.safetensors (c1100)
- eval_t3_results.json (raw loss)
