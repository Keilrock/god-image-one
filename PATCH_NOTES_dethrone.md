# Patch notes — branch `jalur-c-dethrone`

Eksperimen berbasis intel tournament `tourn_43119372e7807746_20260611` (lihat analisa final round).
Branch `jalur-b-sizeaware` (aman buat submit) **tidak disentuh**. Tidak di-push — review & push sendiri.

## ⚠️ Koreksi penting (vs diagnosa awal)
Awalnya disangka `caption_dropout = 0` (dari toml). **Salah.** `scripts/image_trainer.py` meng-**override** ke `0.1` di akhir `create_config` untuk SDXL → toml-nya mati. Jadi:
- SDXL kamu sebenarnya **0.1** (lebih TINGGI dari juara 0.05 → kemungkinan over-regularize).
- Qwen/Z **tidak ada** caption_dropout (ai-toolkit branch tak menyetelnya → 0).

## Yang diubah (auto-applied di branch ini)

| # | File | Perubahan | Alasan | Status |
|---|---|---|---|---|
| 1 | `scripts/image_trainer.py:~335` | SDXL `caption_dropout_rate` **0.1 → 0.05** | Samakan dgn juara Qwen (config.yaml = 0.05); 0.1 over-regularize | 🔬 INFERRED utk SDXL (metadata SDXL juara di-strip; 0.05 kebukti hanya di Qwen) |
| 2 | `scripts/image_trainer.py` `config_mapping` 228/235 | SDXL rank **32 → 64** (alpha 64) | Bos menang product dgn plain-LoRA **rank 64**; rank 32 kamu paling rendah | ✅ rank 64 VERIFIED dipakai pemenang |
| 3 | `scripts/image_trainer.py` `compute_aitoolkit_steps` | per_image 100→**200**, min_steps 600→**2000** | Formula lama hasilkan ~1400 step utk dataset tipikal (~14 img) = **persis angka KALAH challenger**. Floor 2000 → Qwen 2000–3000 (bracket bos 2500), Z=2000 (angka menang bos). | ✅ step menang VERIFIED; floor 2000 fix under-training |
| 4 | `qwen_image.yaml` + `zimage.yaml` dataset | **+ `caption_dropout_rate: 0.05`** | Bocorkan konsep ke unconditional (75% skor = no-text); juara Qwen pakai 0.05 | ✅ 0.05 VERIFIED (Qwen config juara) |

## YANG TIDAK di-apply: DoRA (LyCORIS) — opsional, butuh verifikasi dulu

Pemenang SDXL (challenger) pakai **LyCORIS DoRA** (`network_module = lycoris.kohya`, `dora_wd=True`), dan untuk **style** + `loraplus_lr_ratio=16`. **Tidak gue apply** karena:
- Package `lycoris_lora` **belum kelihatan terinstall** di repo (cuma ada komentar di `sd-script/train_network.py`). Kalau `lycoris.kohya` dipanggil tanpa package → **build/training jebol**.
- Risiko terlalu besar buat di-blind-apply.

**Bukti penting soal DoRA (jangan DoRA semuanya):** di final, **plain-LoRA rank 64 (bos) ngalahin DoRA rank 64 (challenger) di product sebesar 26%**. DoRA cuma menang sah di **style**. Jadi:
- **product → plain LoRA rank 64** (sudah jadi default branch ini). Jangan DoRA.
- **style → DoRA + LoRA+** (kandidat upgrade, lihat di bawah).
- **logo → DoRA** elite tapi opsional.

### Cara enable DoRA (manual, setelah pastikan lycoris terinstall)
1. Tambah ke dockerfile trainer: `pip install lycoris_lora` (atau `prodigy-plus`/lycoris sesuai versi sd-scripts).
2. Di `base_diffusion_sdxl_style.toml`:
   ```toml
   network_module = "lycoris.kohya"
   network_args = ["conv_dim=4","conv_alpha=4","algo=lora","dora_wd=True","loraplus_lr_ratio=16","dropout=0"]
   network_dim = 32
   network_alpha = 32
   ```
3. Test 1 task style lokal dulu — pastikan training jalan & output `.safetensors` ter-load di eval (key `lora_unet_*`).

## ⚠️ Disiplin backend (pelajaran dari blunder bos)
Bos kalah 1 task SDXL karena nyetor **LoRA format DiT/ai-toolkit** (`diffusion_model.transformer_blocks.*`) di task SDXL → LoRA tak ke-load → loss 0.0649. **Pastikan routing `model_type` benar:** SDXL/Flux → kohya (`lora_unet_*`), Qwen/Z → ai-toolkit. Branch ini tidak mengubah routing (sudah benar di repo-mu), tapi jaga jangan sampai regресi.

## Cara validasi sebelum percaya (WAJIB)
Eval deterministik (`master_seed=42`, strength 1.0, `EVAL_DEFAULTS`, empty-prompt utk no-text):
1. Self-split 1 dataset (mis. logo/person) 80/20.
2. Train versi `jalur-b` (lama) vs `jalur-c` (branch ini) di task yang sama.
3. Bandingkan skor `0.25·text + 0.75·no_text` + breakdown text vs no_text.
4. Target: no_text turun (caption_dropout) & overall ≤ versi lama. Kalau caption_dropout 0.05 malah lebih jelek dari 0.1 di SDXL, revert #1.

## Ringkas prioritas riset
1. **person** (Qwen) — bar ~0.086, lemah utk semua → peluang terbesar. Coba: step banyak (2000–2500) + caption_dropout 0.05 + (opsional) rank 128.
2. **konsistensi SDXL ~0.040** — rank 64 + backend benar; jangan kalah tipis.
