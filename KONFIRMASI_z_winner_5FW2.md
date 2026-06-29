# KONFIRMASI recipe Z pemenang 5FW2 dari HF metadata (CPU, no GPU, no train)

> Tujuan: bongkar model Z-Image pemenang 18-Jun (5FW2Eaae) dari HF, konfirmasi recipe PERSIS dari
> metadata LANGSUNG. Nentukan keputusan: batalin OneTrainer DoRA buat Z → ganti ai-toolkit plain LoRA.
> Branch jalur-e-aitoolkit. Cara: Gradients auditing API (cari task) + download config.yaml + baca
> safetensors header via HTTP Range. Gak ada yang diubah/commit.

---

## Repo & task (ketemu)
- **Task ID Z** (via `https://api.gradients.io/auditing/tasks/{id}`): **`956b14d3-7c4d-4d1a-8614-19d6bd025b2f`**
  → model_type `z-image`, base `gradients-io-tournaments/Z-Image-Turbo`, ds `person_evelyn_reed`. ✓
- **Repo HF:** `gradients-io-tournaments/tournament-tourn_c43e1fc22c71be8f_20260618-956b14d3-7c4d-4d1a-8614-19d6bd025b2f-5FW2Eaae`
- Files: `checkpoints/config.yaml` (config training ASLI) + `checkpoints/last.safetensors`.

## Bukti dari safetensors `__metadata__` + tensor (last.safetensors)
| cek | hasil |
|---|---|
| **Trainer** | **`ai-toolkit` v0.7.10** (field `software`) — BUKAN OneTrainer |
| **`dora_scale` tensor?** | **0 (NOL)** → **PLAIN LoRA, BUKAN DoRA** 🔴 KUNCI |
| total tensors / modul | 480 tensors, **240 lora_down (lora_A)** = 240 modul |
| **rank** | lora_A shape `[32, 256]` → **rank 32** |
| key naming | `diffusion_model.layers.N.attention.to_k.lora_A/lora_B.weight` (ai-toolkit `diffusion_model.*`) |
| training_info | step **168**, epoch 3 ; ss_base_model_version zimage |

## Recipe lengkap dari `config.yaml` (ada di repo → LR/step/EMA KE-BACA, bukan ngarang)
```
network:  type: lora, linear 32 / linear_alpha 32, conv 16 / conv_alpha 16   (plain LoRA + conv, no decompose)
train:    steps 168, lr 1e-4, optimizer adamw8bit, noise_scheduler flowmatch, timestep_type weighted, batch 1, bf16
model:    arch zimage:turbo, assistant_lora_path zimage_turbo_training_adapter_v2.safetensors, qfloat8 + quantize_te (fp8)
EMA:      (absent) -> NO EMA
CFG:      (absent) -> NO do_cfg / cfg_scale
caption_dropout: (absent di datasets) -> NONE
dataset:  resolution [512,768,1024], trigger "Evelyn Reed", save diffusers / save_every 250
```

## 🔴 VONIS INTI
**Z pemenang 5FW2 = ai-toolkit PLAIN LoRA (rank 32 + conv 16), BUKAN DoRA.**
Bukti keras: **0 `dora_scale` tensor** + `software: ai-toolkit 0.7.10`. Bongkaran lama jalur-d
(`winner_z_aitoolkit_config.yaml`) **BENAR** — match PERSIS (linear32/conv16/168step/adamw8bit/lr1e-4/no-EMA/no-CFG/trigger Evelyn Reed).

## Ke-baca vs nggak (jujur)
- Ke-baca SEMUA — karena `config.yaml` committed di repo (bukan cuma safetensors __metadata__): network/rank/conv/steps/lr/optimizer/EMA(absent)/CFG(absent)/cd(absent)/fp8/trigger.
- Dari safetensors __metadata__ doang: trainer, dora_scale(none), rank, modul, keys. cd/EMA gak ada di __metadata__ → dikonfirmasi absent via config.yaml.

## Banding vs Qwen person pemenang (5FW2)
| | Z (5FW2) | Qwen person (5FW2) |
|---|---|---|
| network | plain LoRA **32** + conv16 | plain LoRA **128** |
| EMA | **NO** | EMA-on (kita pakai EMA-**off** = 0.06825) |
| CFG | **NO** | do_cfg true, cfg 6.0 |
| steps | 168 | 108 |
| trainer | ai-toolkit | ai-toolkit |

## 🔴 BONUS — pipeline Z template SEKARANG udah ~90% match winner
`base_diffusion_zimage.yaml` HEAD **udah**: plain lora 32 + **conv16** + assistant_lora + qfloat8 + lr1e-4 + adamw8bit + flowmatch + timestep weighted + no-EMA/CFG. **Beda cuma:**
- 🔴 **steps: pipeline 2000 (size-aware floor) vs winner 168** → overtraining parah (SAMA PERSIS problem Qwen-person F2!)
- 🟡 **caption_dropout: pipeline 0.05 vs winner none**

## Konfirmasi arah
**Batalin OneTrainer DoRA + Z ke ai-toolkit plain LoRA = DIDUKUNG DATA + LOW-RISK.**
- Winner Z resmi (beat boss: 0.0401 < 0.0414) = ai-toolkit plain LoRA — terkonfirmasi dari metadata.
- Pipeline udah route Z→ai-toolkit + template udah match network. **Cuma butuh step-fix ala-F2** (Z person → steps ~168, scoped) + cd —
  **TANPA** risiko Dockerfile/OneTrainer/build dari F3 Fase 0.

⚠️ **Pertimbangan jujur:** jalur-d OneTrainer DoRA skor **0.0308** (LEBIH BAGUS dari ai-toolkit 0.0401). Jadi OneTrainer *bukan* salah —
dia lebih bagus, TAPI butuh integrasi berisiko (build bisa matiin Qwen) + step-fit window belum diukur (GATE 1B TBD).
ai-toolkit plain **udah cukup ngalahin boss** (0.0401<0.0414) dengan risiko jauh lebih kecil. Trade-off skor-vs-risiko = keputusan Wen.
Datanya: **winner resmi = ai-toolkit plain LoRA.**

### Sumber/bukti
- Gradients API `auditing/tasks/956b14d3...` (model_type z-image, Z-Image-Turbo, evelyn_reed).
- HF `gradients-io-tournaments/tournament-tourn_c43e1fc22c71be8f_20260618-956b14d3-...-5FW2Eaae`:
  `checkpoints/config.yaml` + `checkpoints/last.safetensors` (__metadata__ software=ai-toolkit, 0 dora_scale, 240 modul, rank32).
- Banding `scripts/core/config/base_diffusion_zimage.yaml` (HEAD) + `jalur-d-perjuangan/winner-recipes/winner_z_aitoolkit_config.yaml`.
