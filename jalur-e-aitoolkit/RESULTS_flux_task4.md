# FASE 6 — FLUX #4 TRAIN + EVAL RESULT (2026-06-27)

## 🏆 VERDICT: NYALIP JUARA & BOSS — recipe template verbatim (rank128/Lion/250)

Train full 250 step (template juara verbatim, base `mhnakif/fluxunchained-dev`), eval pakai validator asli
(`gradientsio/image-evaluator:basilica` + redisfix). Pipeline DIVALIDASI (sanity reproduce 5FW2 & boss PERSIS).

## Setup
- Recipe: `base_diffusion_flux.toml` verbatim (flux.json default udah di-revert ke `{}`). rank128/alpha64,
  Lion (wd0.005), unet_lr 8e-5 / te_lr 8e-6, guidance 85, flow-shift 3.1582, sigmoid, train_t5xxl, 1024, seed 2.
- Env: docker `diagonalge/kohya_latest` (torch2.1.2/diffusers0.32), sd-scripts BUILT-IN image.
- Train: 250 step = epoch 125, ~62 menit (≈ estimasi gate 64 min @14.3s/it). 13 checkpoint (epoch 10-120 + last).
- Eval: validator asli, test set 3fbace30 (2 img), master_seed 42, weighted 0.25*text+0.75*no_text. 40 gen/eval.

## ✅ SANITY (validasi pipeline flux — reproduce angka resmi)
| model | text | no_text | WEIGHTED kita | OFFICIAL 18-Jun | delta |
|-------|------|---------|---------------|-----------------|-------|
| **5FW2** (juara) | 0.03233 | 0.03770 | **0.03636** | 0.036437 | **−0.2%** (persis) |
| **BOSS 5GU4** | 0.03222 | 0.03882 | **0.03717** | 0.037169 | **0.0%** (persis) |
→ Pipeline flux/kohya VALID. Angka kita di bawah ini dipercaya.

## 🎯 HASIL CHECKPOINT KITA
| checkpoint | text | no_text | **WEIGHTED** | vs 5FW2 (0.036437) | vs BOSS (0.037169) |
|------------|------|---------|--------------|--------------------|---------------------|
| epoch 80  | 0.03217 | 0.03671 | 0.03557 | −2.4% | −4.3% |
| epoch 90  | 0.03185 | 0.03644 | 0.03530 | −3.1% | −5.0% |
| epoch 100 | 0.03180 | 0.03639 | 0.03524 | −3.3% | −5.2% |
| **epoch 110** | 0.03099 | 0.03641 | **0.03506** | **−3.8%** | **−5.7%** |
| **epoch 120** (zona verdict, window-reachable) | 0.03095 | 0.03644 | **0.03507** | **−3.8%** | **−5.6%** |
| last (epoch 125, ~step250, bonus >120) | 0.03112 | 0.03637 | 0.03506 | −3.8% | −5.7% |

## Analisis
1. **MENANG di zona verdict (epoch 120, apple-to-apple)**: 0.03507 vs juara 5FW2 0.036437 (**−3.8%**) & boss 0.037169 (**−5.6%**). Checkpoint epoch 120 = yang ke-reach dalam window 1.0h turnamen → **relevan buat submission**.
2. **Kurva**: 80→0.03557, 90→0.03530, 100→0.03524, 110→0.03506, 120→0.03507, 125→0.03506. Turun monoton lalu **flat ~0.0350 mulai epoch 110**. Sweet spot epoch 110-120.
3. **Bonus >120 (epoch 125) = 0.03506** ≈ sama epoch 120 (0.03507). **Lebih dari 120 GAK nambah** (flat). Konfirmasi: walau kita bebas window lokal, gak ada gain di >120 → epoch ~120 emang titik optimal. Submission pilih epoch 120 (atau 110), BUKAN >120 (toh sama, dan di turnamen >120 gak ke-reach).
4. **Kenapa kita > juara walau recipe IDENTIK?** Recipe sama persis (template verbatim, seed 2). Selisih = **non-determinism run-to-run** (bf16+Lion+xformers+hardware non-deterministic walau seed sama). Eval THEIR 5FW2 ckpt = 0.03636, eval OUR epoch-120 = 0.03507 — metrik sama, run beda → kita dapet minimum lebih bagus. Sanity buktiin eval exact, jadi keunggulan GENUINE.

## ⚠️ Caveat jujur
- Ini **1 run**. Variance run-to-run NYATA (recipe sama: kita 0.0351 vs mereka 0.0364). Band variance ~0.035-0.037. Kita landing di ujung bagus. Run lain bisa 0.035-0.037. TAPI epoch 120 kita (0.03507) di bawah boss (0.0372) dengan margin lebar (−5.6%) → aman nyalip walau ada variance.
- Buffer ke target 0.0372 (boss) = **−5.6%**, ke juara 0.0364 = **−3.8%**. Dua-duanya LEWAT.

## REKOMENDASI SUBMISSION
**Checkpoint epoch 120** (`last-000120.safetensors`, 0.03507) = submission #4. Window-reachable, nyalip juara & boss.
Alternatif epoch 110 (0.03506, seri). Jangan >120 (gak ke-reach turnamen, lagian flat).

## Artefak
- /ephemeral/flux_run/out/last-0000{10..120}.safetensors + last.safetensors (3.47GB each, 13 ckpt)
- eval json: /ephemeral/flux_run/eval_*.json
- config: flux_src_ref/config_gate_used.toml + config_full.toml (di repo)
- HF backup: checkpoint kunci → (lihat commit)
