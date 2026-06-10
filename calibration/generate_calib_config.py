#!/usr/bin/env python3
"""
Standalone epoch-calibration config generator (PILOT).

Produces a temporary SDXL training .toml for the epoch-overfit sweep WITHOUT
touching scripts/lrs/*.json. It faithfully replicates production create_config()
(scripts/image_trainer.py) for the chosen (base_model, size-bucket) pair, then
applies the calibration overrides (long run + staged checkpoints + samples).

Recipe basis (decided): data[hash(nova-anime)][s] — the production recipe that
actually ran in the tournament — NOT the generic default[s].

Run on CPU (no GPU needed) just to emit + preview the config:
    python3 calibration/generate_calib_config.py --preview
"""
import argparse
import hashlib
import json
import os
import toml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO, "scripts/core/config/base_diffusion_sdxl_person.toml")
PERSON_CFG = os.path.join(REPO, "scripts/lrs/person_config.json")  # READ ONLY

# ── Pilot target ────────────────────────────────────────────────────────────
TASK_ID = "95f68a11-3b13-488f-8ed5-42fec8865160"
BASE_MODEL = "John6666/nova-anime-xl-pony-v5-sdxl"
BUCKET = "s"  # 14 images -> bucket s (11-20)

# network mapping for this model = 235 (from create_config network_config_person)
NETWORK_235 = {
    "network_dim": 32,
    "network_alpha": 32,
    "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"],
}

# ── Calibration overrides (the ONLY things we vary vs production) ────────────
CALIB_OVERRIDES = {
    "max_train_epochs": 40,      # intentionally long so the overfit knee is visible
    "save_every_n_epochs": 4,    # staged checkpoints at 4,8,...,36 (separate files)
    "sample_every_n_epochs": 4,  # pilot-only: 1 sample grid per checkpoint for visual check
    # save_last_n_epochs intentionally left UNSET (None) -> no pruning, keep all 10
    # base template sets async_upload=true + empty huggingface_repo_id -> every
    # checkpoint save errors on an empty repo push. Calibration uploads separately
    # in PHASE 5, so disable the in-trainer HF push entirely.
    "async_upload": False,
}

# Pilot-only sample prompts (subject trigger from the dataset = "Susie")
SAMPLE_PROMPTS = [
    "A happy Susie smiling at the camera, standing in front of a white house with a garden. --n blurry, lowres, deformed --w 1024 --h 1024 --s 28 --l 7 --d 42",
    "Susie laughing with friends in a sunlit room --n blurry, lowres, deformed --w 1024 --h 1024 --s 28 --l 7 --d 42",
    "portrait of Susie, neutral expression, plain background --n blurry, lowres, deformed --w 1024 --h 1024 --s 28 --l 7 --d 42",
]


def hash_model(model: str) -> str:
    return hashlib.sha256(model.encode("utf-8")).hexdigest()


# Diffusers component sub-folders. A weights file living inside one of these is a
# single component (e.g. text_encoder/model.safetensors), NOT a full SDXL model.
_COMPONENT_DIRS = {
    "unet", "vae", "text_encoder", "text_encoder_2",
    "tokenizer", "tokenizer_2", "scheduler",
}


def validate_model_path(model_path: str) -> None:
    """Reject the class of bug that produced an all-UNet-missing load.

    Valid pretrained_model_name_or_path is either:
      * a Diffusers FOLDER (dir containing model_index.json) — sd-scripts loads it
        via StableDiffusionXLPipeline.from_pretrained and converts the UNet, or
      * a full single-file SDXL checkpoint (.safetensors / .ckpt).

    A weights file nested in a component dir (text_encoder/, unet/, ...) is NOT a
    full model and silently yields a broken UNet. Caught here by name, so it fails
    even during CPU --preview where the file may be absent.
    """
    parent = os.path.basename(os.path.dirname(model_path.rstrip("/")))
    if parent in _COMPONENT_DIRS:
        raise SystemExit(
            f"[fatal] model_path points at a Diffusers component ('{parent}/'), "
            f"not a full model: {model_path}\n"
            f"        Pass the Diffusers folder (dir with model_index.json) "
            f"or a full single-file SDXL .safetensors."
        )
    if os.path.isdir(model_path):
        if not os.path.isfile(os.path.join(model_path, "model_index.json")):
            raise SystemExit(
                f"[fatal] Diffusers folder is missing model_index.json: {model_path}"
            )
    elif os.path.exists(model_path) and not model_path.endswith((".safetensors", ".ckpt")):
        raise SystemExit(
            f"[fatal] model_path is neither a Diffusers folder nor a checkpoint file: {model_path}"
        )


def build_config(model_path: str, train_data_dir: str, output_dir: str,
                 output_name: str, sample_prompts_path: str) -> dict:
    # 1) start from the SAME base template production uses
    with open(TEMPLATE) as f:
        config = toml.load(f)

    # 2) apply the production size-bucket settings: data[hash][bucket]
    person = json.load(open(PERSON_CFG))
    h = hash_model(BASE_MODEL)
    assert h in person["data"], f"{BASE_MODEL} hash not in person_config data"
    bucket_settings = person["data"][h][BUCKET]
    for k, v in bucket_settings.items():
        config[k] = v

    # 3) paths + network dims, exactly like create_config() does
    validate_model_path(model_path)
    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = train_data_dir
    config["output_dir"] = output_dir
    config["output_name"] = output_name
    config["network_dim"] = NETWORK_235["network_dim"]
    config["network_alpha"] = NETWORK_235["network_alpha"]
    config["network_args"] = NETWORK_235["network_args"]

    # 4) production hardcodes caption_dropout_rate=0.1 AFTER applying lrs settings
    #    (image_trainer.py:307) -> this overrides bucket's 0.05. Replicate faithfully.
    config["caption_dropout_rate"] = 0.1

    # 5) calibration overrides (epochs + staged checkpoints + samples)
    for k, v in CALIB_OVERRIDES.items():
        config[k] = v
    config["sample_prompts"] = sample_prompts_path  # template default was ""
    # sample_sampler already "euler_a" in template

    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=f"<GPU: cache path or .safetensors for {BASE_MODEL}>",
                    help="local base-model path resolved on GPU (training_paths.get_image_base_model_path)")
    ap.add_argument("--train-data-dir", default=f"/workspace/calibration/data/{TASK_ID}/img",
                    help="parent dir containing the '5_lora style' repeat-subfolder")
    ap.add_argument("--output-dir", default=f"/workspace/calibration/checkpoints/{TASK_ID}_nova_s_calib")
    ap.add_argument("--output-name", default="nova_s_calib")
    ap.add_argument("--out", default=os.path.join(REPO, "calibration/out", f"{TASK_ID}_nova_s_calib.toml"))
    ap.add_argument("--preview", action="store_true", help="print the final toml to stdout")
    args = ap.parse_args()

    # ABSOLUTE so it resolves regardless of the trainer's cwd (the container runs
    # with cwd=scripts/sd-script; a relative path silently yields zero samples).
    out_abs = os.path.abspath(args.out)
    sample_prompts_path = os.path.join(os.path.dirname(out_abs), "sample_prompts.txt")
    with open(sample_prompts_path, "w") as f:
        f.write("\n".join(SAMPLE_PROMPTS) + "\n")

    config = build_config(args.model_path, args.train_data_dir, args.output_dir,
                          args.output_name, sample_prompts_path)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        toml.dump(config, f)

    print(f"[ok] wrote {args.out}")
    print(f"[ok] wrote {sample_prompts_path}")
    if args.preview:
        print("\n===== FINAL CALIBRATION CONFIG (preview) =====\n")
        print(toml.dumps(config))


if __name__ == "__main__":
    main()
