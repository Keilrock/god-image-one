#!/usr/bin/env python3
"""
Upload each LoRA checkpoint to its OWN HF repo so the validator scorer can fetch them.

Why one repo per checkpoint: the validator's find_latest_lora_submission_name() returns
only the highest-epoch file per repo. To score all 10 epochs we give it 10 repos.

Layout each repo must have (validator expects):
  checkpoint/ckpt-<NNNNNN>.safetensors      (DIFFUSION_HF_DEFAULT_FOLDER = "checkpoint")

sd-scripts emits in the checkpoints dir:
  nova_s_calib-000004.safetensors ... nova_s_calib-000036.safetensors   (epochs 4..36)
  nova_s_calib.safetensors                                              (final epoch = max_train_epochs=40)

Repos are created PUBLIC so the eval container can pull without a token.
Run on GPU phase (needs HF login). Writes repos.json (repo_id -> epoch) for parsing.
"""
import argparse
import json
import os
import re

from huggingface_hub import HfApi


def epoch_of(filename: str, max_epoch: int) -> int:
    m = re.search(r"[-_](\d+)\.safetensors$", filename)
    return int(m.group(1)) if m else max_epoch  # bare name = final epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir with nova_s_calib*.safetensors")
    ap.add_argument("--hf-user", required=True, help="HF username/org to create repos under")
    ap.add_argument("--prefix", default="calib-nova-s-ep", help="repo name prefix")
    ap.add_argument("--max-epoch", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "repos.json"))
    ap.add_argument("--private", action="store_true", help="create private repos (then eval needs HF_TOKEN)")
    args = ap.parse_args()

    api = HfApi()
    files = sorted(f for f in os.listdir(args.ckpt_dir) if f.endswith(".safetensors"))
    assert files, f"no .safetensors in {args.ckpt_dir}"

    mapping = {}
    for fn in files:
        ep = epoch_of(fn, args.max_epoch)
        repo_id = f"{args.hf_user}/{args.prefix}{ep:02d}"
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=os.path.join(args.ckpt_dir, fn),
            path_in_repo=f"checkpoint/ckpt-{ep:06d}.safetensors",
            repo_id=repo_id,
            repo_type="model",
        )
        mapping[repo_id] = ep
        print(f"[ok] epoch {ep:>2} : {fn} -> {repo_id} (checkpoint/ckpt-{ep:06d}.safetensors)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[ok] {len(mapping)} repos -> {args.out}")
    print("MODELS=" + ",".join(sorted(mapping, key=lambda r: mapping[r])))


if __name__ == "__main__":
    main()
