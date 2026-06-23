# GATE 1 RESULT — jalur-d-perjuangan (OneTrainer + DoRA for Qwen/Z)

**Date:** 2026-06-23 · **Test rig:** NVIDIA L40 48GB (driver 535 / CUDA 12.2) · **Verdict: 1A PASS → PROCEED to Fase 2 (H100); 1B step-fit within window = TBD (must measure on H100)**

DoRA-via-OneTrainer is **alive for both Qwen and Z-Image**. The critical single-point-of-failure (does OneTrainer's DoRA key format map in the validator's ComfyUI?) PASSED cleanly for both. **No musubi+LoKr fallback needed.**

---

## TES 1A — ComfyUI key-map load-test (CRITICAL)

Method: trained a tiny DoRA (decompose=on, FP8, rank 16, dummy 12-img dataset, ~24–204 steps — quality irrelevant) per model, then loaded it with the **validator's own ComfyUI code**, run **inside the validator container** `gradientsio/image-evaluator:basilica` (exact eval env). Built the lora key-map exactly as the validator's LoraLoader does (`comfy.lora.model_lora_keys_unet`) and applied the DoRA via `comfy.lora.load_lora` @ strength 1.0, counting unmatched keys.

| model | comfy model class | key_map entries | DoRA modules | patches loaded | `lora key not loaded` | verdict |
|---|---|---|---|---|---|---|
| **Qwen-Image** | `QwenImage` | 6281 | 720 | **720** | **0** | **PASS ✅** |
| **Z-Image** | `Lumina2` | 2043 | 210 | **210** | **0** | **PASS ✅** |

`dora_scale` IS applied by comfy's `weight_adapter` (`weight_decompose`) — DoRA decomposition honored, not silently dropped.

### Key-map: Qwen vs Z (naming differs, both map)
OneTrainer saves keys as `transformer.<...>.{lora_down.weight, lora_up.weight, alpha, dora_scale}`:
- **Qwen:** `transformer.transformer_blocks.0.attn.add_k_proj.*` (720 modules → 2880 tensors)
- **Z:** `transformer.layers.0.attention.to_k.*` (210 modules → 840 tensors)

Despite different internal naming, comfy's `model_lora_keys_unet` maps **100%** of modules for both. The plan's two caveats (DoRA runs on Qwen/Z specifically; `dora_scale` loads in the standard ComfyUI loader) are now **empirically confirmed end-to-end**, not assumed.

---

## TES 1B — wall-clock & training-window fit

**1A is GPU-independent and stands (PASS). 1B re-scoped after verifying the validator's actual image-task window.**

### Validator image training window — ACTUAL (corrects an earlier error)
> ~~Earlier draft said "1250 steps fits within `MAX_TRAINING_HOURS = 6.0`" — that was WRONG for image.~~ `MAX_TRAINING_HOURS = 6.0` (`validator/core/constants.py:75`) is the **text/instruct/GRPO** throughput-budget cap: `TASK_TYPE_HOURS_MULTIPLIER` (`constants.py:101`) holds only `INSTRUCTTEXTTASK/CHATTASK/DPOTASK` — **no `IMAGETASK`** — and the GRPO band ends at `(inf, 6.0)`. **It does NOT apply to image tasks.**

Real image window = per-task `hours_to_complete`, set in `validator/tasks/diffusion_synth.py` (`_random_image_competition_hours()` line 589, `create_synthetic_image_task()` lines 594-602) from `validator/core/constants.py:176-178`:

| model | `hours_to_complete` (random per-task, 15-min steps) | extra |
|---|---|---|
| **Qwen-Image** | **1.0 / 1.25 / 1.5 h** | base {0.5,0.75,1.0} + `QWEN_IMAGE_EXTRA_COMPETITION_HOURS` (+0.5h) |
| **Z-Image** | **0.5 / 0.75 / 1.0 h** | none |
| SDXL / Flux | 0.5 / 0.75 / 1.0 h | none |

- **Random per task** (`MIN_IMAGE_COMPETITION_HOURS=0.5` .. `MAX_IMAGE_COMPETITION_HOURS=1.0`); **hard-enforced** — container marked FAILURE + killed at `started_at + hours_to_complete + 10 min grace` (`STALE_TASK_GRACE_MINUTES`, `trainer/utils/cleanup_loop.py:42-50`).
- Window = **TOTAL job time**: container start + base-model download (~54GB for Qwen) + training + checkpoint upload. **Actual training time < window.**
- Only `QWEN_IMAGE` gets +0.5h. `ImageModelType` = {FLUX, SDXL, Z_IMAGE, QWEN_IMAGE} are distinct — **Z gets no extra** despite also being ai-toolkit (split is per-model, not per-backend). Tournament/boss-round reuses the same function (`validator/tournament/task_creator.py:387,400`).

### Step budget per window = **TBD — MUST measure on H100 (Fase 2)**
L40 measured ~15 s/step (Qwen; FP8-storage + GPU-checkpoint + compile) / ~1.3–3 s/step (Z). **This is L40, not H100 — L40→H100 extrapolation is NOT reliable for decisions, so no step count is claimed here.** In Fase 2, measure real Qwen/Z H100 throughput + base-download overhead (~54GB), then compute how many steps fit the **worst-case** window (Qwen 1.0h / Z 0.5h). **Risk to check: undertrained if effective steps < boss (~1250 Qwen).**

**1B verdict:** DoRA feasible + ComfyUI load PASS (1A); **step-fit within window = TBD on H100.**

---

## 3 catatan buat Fase 2 (reproduce di H100)

1. **torch / driver:** OneTrainer pins `torch 2.12.0+cu130` (needs driver ≥580). The L40 VM (driver 535 / CUDA 12.2) can't run cu130, so the gate used **`torch 2.11.0+cu128`** (CUDA 12.x minor-version compat) + a small shim in `adamw_extensions.py`/`adam_extensions.py` (`_cuda_graph_capture_health_check` → `_accelerator_graph_capture_health_check`). **On H100 with driver ≥580, use the pinned `torch 2.12.0+cu130` directly — the shim is then unnecessary.**
2. **FP8 = storage-only:** OneTrainer `FLOAT_8` quantizes weight *storage* (compute upcasts to bf16) → no fp8 tensor-core speedup on L40. H100 speedup comes from raw bf16 FLOPS, not fp8 matmul.
3. **DoRA = linear-only on Qwen/Z DiT:** no conv2d DoRA (DiT has no conv to decompose) — matches the plan's "DoRA-linear" expectation, edge thinner than the SDXL win (which included conv4).

---

## Config that works (committed verbatim in `onetrainer-configs/`)
- `config_qwen_dora.json` — Qwen DoRA (`__version:10`, `peft_type:LORA` + `lora_decompose:true`, `transformer.weight_dtype:FLOAT_8`, `gradient_checkpointing:ON`, `compile:true`).
- `config_z_dora.json` — Z DoRA (model_type `Z_IMAGE`, Z-specific `layer_filter` regex + `quantization.layer_filter:"layers"`, backups OFF).
- `concepts.json` — dummy concept. `loadtest.py` — the faithful comfy key-map test.

Config quirks learned: must set `"__version": 10` (else legacy migration breaks the `optimizer` dict); `backup_after:0` means backup **every step** (set `backup_after_unit:"NEVER"`); `training_samples/samples.json` must exist (`[]`).

## Artifacts (GATE proof) — on HuggingFace
- `dora_qwen.safetensors` (219MB) + `dora_z.safetensors` (72MB)
- **HF repo:** https://huggingface.co/keilrockstars/god-image-dora-gate (private)

## Scope honored
jalur-c (e185164) untouched · SDXL/kohya recipe untouched · no full sweep · only Qwen/Z branch.
