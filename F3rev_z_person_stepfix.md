# F3-REVISED — Z person step-fix ke pipeline (ai-toolkit plain, BUKAN OneTrainer)

> Tujuan: menangin Z via ai-toolkit plain LoRA (recipe pemenang 5FW2), OneTrainer DIBATALKAN.
> Z pemenang = plain LoRA-32+conv16, steps 168 (=12×14img), no-EMA/CFG/cd. Template Z pipeline udah 90% match —
> CUMA steps (2000→12×img) + cd yang salah. Extend pola F2 (Qwen person) ke Z person.
> CPU only, no GPU/train. Branch jalur-e-aitoolkit. Commit lokal `4544ea3` (belum push).
> Patokan: KONFIRMASI_z_winner_5FW2.md, BANDING_z_5FW2_vs_boss_5GU4.md.

---

## TUGAS 1 — Scope Z person step-fix (`scripts/image_trainer.py`, create_config ai-toolkit branch)
Generalisasi blok F2 jadi handle Qwen **+** Z:
- Deteksi kategori (reuse `scripts/core/category_detector.py` F2) buat **qwen-image DAN z-image** → `ait_category`.
- `qwen_person = (qwen-image AND category==person)`, `z_person = (z-image AND category==person)`,
  `person_12ximg = qwen_person or z_person`.
- **STEP:** `person_12ximg` → `steps = 12 × jumlah_gambar` (else: size-aware existing, Qwen cap3000 / Z cap2000).
- **caption_dropout:** `person_12ximg` → buang (`pop`) — Qwen + Z (pemenang person TANPA cd).
- **EMA-off:** `if qwen_person` DOANG. **Z TIDAK disentuh** (Z pemenang bawaan no-EMA).
- Network / linear / conv / alpha / assistant_lora / fp8 / lr / optimizer: **TIDAK disentuh** (template Z udah match pemenang).

🔴 Beda Z vs Qwen (tidak ketuker):
| field | Qwen person (F2) | Z person (F3-rev) |
|---|---|---|
| network | plain LoRA 128 | plain LoRA 32 + conv16 (template, utuh) |
| EMA | di-set false | **none — tidak di-set** |
| CFG | ada (template) | none (tidak ditambah) |
| caption_dropout | buang | buang |
| steps | 12×img (108=12×9) | 12×img (168=12×14) |
| assistant_lora | — | Turbo v2 (utuh, tidak disentuh) |

→ Z person CUMA 2 ubahan runtime: **steps (12×img)** + **buang caption_dropout**. Lebih minimal dari Qwen.

## TUGAS 2 — Test create_config (CPU) — ALL PASS ✅
| scenario | steps | ema | cd | cfg | network | assist |
|---|---|---|---|---|---|---|
| **Z PERSON** (14img, Evelyn) | **168** (12×14) | **ABSENT** | **ABSENT** | ABSENT | lora lin32/conv16 | yes |
| Z LOGO (36img, BrandEssence) | 2000 (size-aware) | absent | 0.05 | absent | lin32/conv16 | yes |
| QWEN PERSON (9img, David — F2) | 108 | **False** | ABSENT | True | lin128 | — |
| Z ART (no trigger, Dreamlike) | 2000 (size-aware) | absent | 0.05 | absent | lin32/conv16 | yes |
| SDXL (branch toml) | (toml) | n/a | 0.05 | — | networks.lora | — |

Tegasan:
- ✅ **Z person**: 12×img (168) + no-cd, TAPI **network/conv/EMA-none/assistant_lora UTUH** (cuma step+cd berubah = persis pemenang 5FW2).
- ✅ **Qwen person F2 GAK ke-rusak** (masih EMA-off + 108 + no-cd).
- ✅ **Z logo/art** → else (size-aware 2000), network utuh — tidak kena fix person.
- ✅ **SDXL/Flux** → branch toml, utuh.

(12 assertion lulus semua, termasuk: Z person steps==168, cd dibuang, EMA tidak di-set, network 32/conv16, assistant_lora ada, no CFG; Z logo size-aware; Qwen F2 utuh; SDXL toml utuh.)

## TUGAS 3 — Template Z verify
`base_diffusion_zimage.yaml` HEAD udah match pemenang: `type lora` lin32/alpha32 + **conv16/conv_alpha16** ✅,
`assistant_lora_path zimage_turbo_training_adapter_v2` ✅, qfloat8 + quantize_te ✅, lr1e-4 / adamw8bit / flowmatch /
timestep weighted ✅, **NO ema_config**, **NO do_cfg** ✅. **Template TIDAK perlu diubah** — code yang handle steps (→12×img) + cd-drop. Tidak ada anomali.

## Catatan & status
- `py_compile` OK. Beda dari Qwen: Z **lebih minimal** (cuma steps + cd, NO EMA).
- 🔴 Re-validate GPU nanti: konfirmasi 168-step Z beneran menang (vs boss overtrain), step final (window-cut), + caption LLaVA-augmented.
- OneTrainer DoRA (0.0308 > 0.0401) DIBATALKAN — ai-toolkit plain udah cukup ngalahin boss (0.0401<0.0414), risiko OneTrainer (matiin Qwen / driver / build) tidak sebanding.
- Commit lokal `4544ea3` — **belum push** (nunggu instruksi Wen).

### File diubah
- `scripts/image_trainer.py` (create_config ai-toolkit branch: blok [F2/F3] kategori + step + cd; EMA-off tetap Qwen-only).
