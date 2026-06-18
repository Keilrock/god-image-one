# DETHRONE ANALYSIS — temuan & lubang (SN56 image, final 11-Jun)

Analisa strategis di balik runbook. Boss = `5GU4Xkd3` (= EMISSION_BURN_HOTKEY). Threshold dethrone = **3% FIXED** (challenger `loss ≤ boss×0.97`; menang 4/6 task = dethrone).

## Hasil final round: BOSS 4–2 CHALLENGER (5Ca32)
| task | kat | backend | boss | chal | menang | margin |
|---|---|---|---|---|---|---|
| T1 | person | Qwen | 0.08585 | 0.08671 | BOSS | 0.9% hinge |
| T2 | logo | Z-Image | 0.04083 | 0.04591 | BOSS | 12.5% |
| T3 | product | SDXL Tempest | 0.04023 | 0.05056 | BOSS | 25.7% |
| T4 | style | SDXL Fluently | 0.03890 | **0.03652** | CHAL | 6.1% (sah) |
| T5 | logo | SDXL protovision | 0.04106 | 0.04127 | BOSS | 0.5% hinge |
| T6 | logo | SDXL protovision | 0.06490 | **0.02260** | CHAL | 65% (boss BLUNDER, LoRA mati) |

Challenger menang sah cuma **1** (T4 style); T6 = boss salah-backend (DiT LoRA di SDXL → key mismatch → mati).

## Bobot kategori (TERVERIFIKASI dari `IMAGE_SYNTH_CATEGORY_WEIGHTS`)
`style .25 · person .25 · logo .15 · social .15 · design .10 · product .10`
→ **person & style = 50%** (paling sering). logo & social .15. design & product .10.
→ Final 11-Jun (3 logo, 0 social/design) = **draw anomali (logo-heavy)**. Final tipikal = person/style-heavy.

## LUBANG / TEMUAN (urut dampak)

**#1 — MATCHING BOSS = TIE = BOSS MENANG.** jalur-c sekarang **persis = boss** di: Z-logo (linear32+conv16+2000), Qwen-person (linear128+2500+TEoff), product (plain-64). Threshold 3% → seri = kalah. Cuma **style** win jelas. → Sweep cuma berarti kalau nemu **edge >3% nyata**, bukan paritas.

**#2 — EV: pecahin PERSON, bukan logo.** person .25 (sering) + sekarang pasti-kalah = swing terbesar. style .25 udah win. logo cuma .15 + draw 11-Jun anomali. → Prioritas: person (FASE 2), khususnya **Qwen TE-on**.

**#3 — T6 "win" = blunder boss, NGGAK repeatable.** Final baru boss nggak submit LoRA mati → boss logo ≈ 0.041 (kaya T5). Semua logo harus ngalahin ~0.041 by 3%. Jangan hitung T6 gratis.

**#4 — Qwen-person TE-on = lever yang boss & chal BELUM coba.** Dua-duanya TE-off. Di fortress paling sering → kandidat edge paling menjanjikan. Taruhan utama (FASE 2B).

**#5 — Test set kecil (1–5 img) → noise.** Flip 3% di 3 img bisa noise. Mitigasi: re-validasi arm menang di task kategori-sama kedua (T5→T6) atau gabung test (T5+T6=6 img). Baca DELTA antar-arm, bukan absolut.

**#6 — Logo butuh RENDER TEKS (25% text-guided).** cd tinggi nyakitin text-following logo; person malah mau cd tinggi (no_text 75%). Global cd 0.05 = kompromi → sweep per-kategori (arm `logo cd0`, `person cd10`).

**#7 — Backend guard = win gratis.** `check_backend_keys.py` sebelum tiap submit → nggak akan kehilangan task gara-gara LoRA mati (mode gagal boss T6). kohya↔SDXL/Flux (`lora_unet_*`), ai-toolkit↔Qwen/Z (`diffusion_model.*`).

**#8 — Z-logo & product = fortress tie (conceded).** jalur-c = boss. Env-hook dukung sweep ai-toolkit kalau mau kejar Z-logo, tapi EV rendah (.15/.10).

## Recipe boss (reverse-engineered, referensi)
- T1 Qwen person: ai-toolkit linear128, 2500 step, **TE off**, cd 0.05, lr1e-4.
- T2 Z logo: ai-toolkit linear32 + **conv16**, 2000 step, lr1e-4.
- T3 SDXL product: kohya **plain `networks.lora` rank 64**.
- T6 SDXL logo: ai-toolkit config di task SDXL → BLUNDER.
- Profil boss = **konsisten ~0.04 di mana-mana** (menang lewat konsistensi, bukan spike).

## Recipe challenger (yang KALAH 4-2, jangan ditiru buta)
- person QF (SDXL): LyCORIS DoRA rank64+conv4, ~65 epoch, TE on.
- T3 product: DoRA-64 → **kalah 26%** (boss plain-64). → bukti **product = plain-64, JANGAN DoRA**.
- T4 style: DoRA32+conv4+**LoRA+ratio16** → menang.
- T6 logo: DoRA32+conv4 → 0.0226 (boss blunder).

## Math dethrone (realistis)
Butuh 4/6 OUTRIGHT win. Foundation: **style (win)**. Fortress (concede): product, Qwen-person (kecuali TE-on tembus). 
→ Jalur paling mungkin: **style + pecahin person (×~1.5 freq)** = ~3 expected, + 1 dari logo/social. **Person = linchpin.** Kalau person tetap fortress, dethrone sangat sulit (style + remah).

## jalur-c default vs boss (status)
| kategori | jalur-c | vs boss |
|---|---|---|
| style (SDXL) | DoRA32+conv4+LoRA+16 | **WIN** (recipe pemenang T4) |
| logo (SDXL) | DoRA32+conv4 | tie/tipis-kalah → FASE 1 cari edge |
| person (SDXL) | plain-64 cd05 | tie → FASE 2A |
| person (Qwen) | linear128+2500+TEoff cd05 | **= boss (tie)** → FASE 2B TE-on |
| logo (Z) | linear32+conv16+2000 cd05 | **= boss (tie)** |
| product (SDXL) | plain-64 | **= boss (tie)** |
| social/design | default plain-64 / DoRA | (gak muncul di final 11-Jun) |
