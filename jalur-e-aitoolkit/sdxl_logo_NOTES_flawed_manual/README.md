# ⚠️ CONFIG MANUAL FLAWED — JANGAN DIPAKE SBG RECIPE AKTIF

Config2 di folder ini = eksperimen B2 logo yang pakai config MANUAL (SALAH):
- `caption_dropout_rate = 0` ❌ (pipeline asli kita 0.05, boss 0.1 — KELEWAT)
- Gak lewat pipeline asli (gak ada LLaVA auto-caption, network hardcode).
- Hasil overfit (best 0.0557 vs boss 0.0465).

NOTES doang buat referensi audit. Recipe LOGO yang BENER harus VIA PIPELINE ASLI (`create_config()`).
Lihat HANDOFF_jalur-e_2026-06-27.md + AUDIT_TODO.md + RESULTS_sdxl_logo.md.
