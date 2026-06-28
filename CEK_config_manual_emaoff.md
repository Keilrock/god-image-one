# CEK Config Manual EMA-off Qwen person — verifikasi lengkap

> Audit config-only (NO GPU, NO train, baca file doang). Tujuan: tahu PERSIS isi config manual
> eksperimen EMA-off Qwen person yang ngasih **0.06825** (sebelum di-timpa ke pipeline default).
> Branch jalur-e-aitoolkit. Patokan: DIFF_qwen_person.md, GIT_HISTORY_MAP.md F2.

**File:** `jalur-e-aitoolkit/configs/ema_off_config.yaml` (ke-commit). Header config-nya nyatain:
*"winner recipe 5FW2Eaae VERBATIM, satu-satunya beda `ema_config.use_ema=false`"*.

---

## 1. Isi lengkap config manual (semua field, apa adanya)

| Field | Value |
|---|---|
| job | extension |
| config.name | `ema_off` |
| **model.arch** | qwen_image |
| model.name_or_path | /cache/models/gradients-io-tournaments--Qwen-Image |
| model.quantize | true |
| model.qtype | float8 |
| model.quantize_te | true |
| model.low_vram | true |
| **network.type** | lora |
| **network.linear** | 128 |
| **network.linear_alpha** | 128 |
| **train.steps** | **108** |
| train.batch_size | 1 |
| train.lr | 0.0001 |
| train.optimizer | adamw8bit |
| train.optimizer_params.weight_decay | 1.0e-05 |
| train.noise_scheduler | flowmatch |
| train.timestep_type | weighted |
| **train.ema_config.use_ema** | **false** ← satu-satunya beda dari winner |
| train.ema_config.ema_decay | 0.995 (moot, EMA off) |
| train.do_cfg | true |
| train.cfg_scale | 6.0 |
| train.dtype | bf16 |
| train.gradient_checkpointing | true |
| train.gradient_accumulation | 1 |
| train.train_unet | true |
| train.train_text_encoder | false |
| train.cache_text_embeddings | true |
| **datasets.caption_dropout_rate** | **TIDAK ADA** (tidak di-set) |
| datasets.resolution | **[512, 768, 1024]** (multi-res) |
| datasets.caption_ext | txt |
| datasets.cache_latents_to_disk | true |
| datasets.is_reg | false |
| datasets.folder_path | /ephemeral/datasets/qwen_t2_train/train_data |
| **trigger_word** | David Miller |
| save.save_every | **36** |
| save.max_step_saves_to_keep | 6 |
| save.save_format | diffusers |
| save.dtype | bf16 |
| training_folder | /ephemeral/output |
| meta.name | qwen_image_lora_ema_off |

---

## 2. Banding config manual (C) vs winner 5FW2 (B)

Sumber B: `jalur-d-perjuangan/winner-recipes/winner_qwen_aitoolkit_config.yaml`.

| Field | Manual EMA-off (C) | Winner 5FW2 (B) | Sama/Beda |
|---|---|---|---|
| **use_ema** | **false** | **true** | 🔴 BEDA (perubahan sengaja) |
| **steps** | **108** | **108** | ✅ SAMA |
| save_every | 36 | 250 | 🟡 beda (granularitas checkpoint) |
| max_step_saves_to_keep | 6 | 4 | 🟡 kosmetik |
| caption_dropout_rate | tidak ada | tidak ada | ✅ SAMA (dua-duanya TANPA cd) |
| network type/linear/alpha | lora 128/128 | lora 128/128 | ✅ |
| lr / optimizer / weight_decay | 1e-4 / adamw8bit / 1e-5 | sama | ✅ |
| noise_scheduler / timestep_type | flowmatch / weighted | sama | ✅ |
| do_cfg / cfg_scale | true / 6.0 | sama | ✅ |
| dtype / qtype / quantize / quantize_te | bf16 / float8 / true / true | sama | ✅ |
| resolution | [512,768,1024] | [512,768,1024] | ✅ |
| train_unet / train_text_encoder | true / false | sama | ✅ |
| cache_text_embeddings / cache_latents | true / true | sama | ✅ |
| trigger_word | David Miller | David Miller | ✅ |

→ **Identik 100% kecuali `use_ema` (off vs on)** + granularitas save (36 vs 250). steps SAMA (108).
Header config benar: cuma EMA yang diubah. (save_every:36 sengaja, biar checkpoint early ke-capture.)

---

## 3. Konfirmasi caption_dropout
✅ **TIDAK di-set** di config manual (tidak ada `caption_dropout_rate` di blok `datasets`).
→ Skor menang **0.06825 dihasilkan TANPA cd eksplisit** (pakai default ai-toolkit).
⚠️ **Berbeda dari template pipeline HEAD** yang paksa `caption_dropout_rate: 0.05` (lever 🟡 di DIFF_qwen_person.md).

---

## 4. Konfirmasi STEPS — config 108, checkpoint terbaik c36

- **Config nge-set `steps: 108`** (BUKAN 36).
- **`save_every: 36`** → checkpoint tersimpan di step **36, 72, 108**.
- Hasil eval (`eval_emaoff_results.json` + `RESULTS_emaoff.md`, test set 106ffec9 hold-out, master_seed 42,
  weighted = 0.25·text + 0.75·no_text):

| ckpt | step | text | no_text | weighted | vs winner 0.07307 |
|---|---|---|---|---|---|
| **c36** | **36** | 0.06275 | 0.07009 | **0.06825** | **−6.6% (TERBAIK)** |
| c72 | 72 | 0.06735 | 0.07366 | 0.07209 | −1.3% |
| c108 | 108 | 0.06359 | 0.07319 | 0.07079 | −3.1% |
| winner (reproduce) | 108 | 0.05375 | 0.07950 | 0.07307 | — |
| boss | — | — | — | 0.0867 | — |

→ **Run jalan sampai 108, tapi checkpoint TERBAIK = c36 (step 36)**. Non-monoton; early-checkpoint malah
terbaik (9-img, less overfit). c36, c72, c108 **semuanya MENANG** vs winner & boss.

---

## ⚠️ Implikasi buat rencana timpa-ke-pipeline (koreksi/penajaman DIFF_qwen_person.md)

1. **Best ≠ final.** Skor terbaik (0.06825) ada di **step 36**, tapi config jalan sampai 108. Di turnamen,
   validator nge-eval checkpoint **FINAL** yang di-upload ke `/app/checkpoints/{task_id}/{repo_name}` (hasil
   akhir run), **bukan** checkpoint early sembarang.
   - Pipeline `steps:108` → final = **c108 = 0.07079** (masih MENANG −3.1%, tapi bukan terbaik).
   - Mau **c36 = 0.06825** (terbaik) → pipeline harus `steps:36` (biar final run = c36).
2. Keputusan timpa: **target steps = 36** (terbaik, tapi early/risiko fluke 9-img) **vs 108** (sesuai boss,
   final c108, lebih known-good, margin lebih kecil). Dua-duanya menang. **Bukan keputusan audit.**
3. Caption_dropout: untuk replikasi PERSIS kondisi 0.06825, **jangan** pakai cd0.05 template — config menang TANPA cd.
4. Ingat (dari DIFF_qwen_person.md): steps di pipeline **tidak bisa** di-set via template doang — `create_config`
   nge-floor ke 2000 (size-aware). Mau 36 atau 108, **wajib ubah kode** image_trainer.py, dan **scoped ke Qwen person**
   biar gak ngerusak Qwen-art (EMA-on) & Z.
