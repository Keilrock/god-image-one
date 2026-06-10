#!/usr/bin/env python3
"""
Deterministic train/test split for the epoch-calibration pilot.

Source: tasks_backup.tar.gz (HF dataset keilrockstars/tourn-data), task 95f68a11
(John6666/nova-anime-xl-pony-v5-sdxl, 14 image+caption pairs, "Susie").

Splits the 14 pairs into:
  - train (11) -> {out_root}/{TASK_ID}/img/5_lora style/   (sd-scripts DreamBooth layout)
  - test  (3)  -> {out_root}/{TASK_ID}/test/                (flat png+txt; validator format)

Deterministic (seed=42, fixed). Re-running gives the identical split. Stays in
bucket s (train 11 in 11-20). DO NOT raise n-test above 3 (train 10 => bucket xs).

Runnable on CPU; no GPU needed.
"""
import argparse
import json
import os
import random
import shutil
import tarfile

TASK_ID = "95f68a11-3b13-488f-8ed5-42fec8865160"
REPEATS = 5            # DIFFUSION_SDXL_REPEATS
INSTANCE = "lora"      # DIFFUSION_DEFAULT_INSTANCE_PROMPT
CLASS = "style"        # DIFFUSION_DEFAULT_CLASS_PROMPT
TRAIN_SUBDIR = f"{REPEATS}_{INSTANCE} {CLASS}"  # -> "5_lora style"


def resolve_tar() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download("keilrockstars/tourn-data", "tasks_backup.tar.gz", repo_type="dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=None, help="path to tasks_backup.tar.gz (default: resolve from HF cache)")
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test", type=int, default=3, help="held-out test count (MAX 3: train must stay >=11 for bucket s)")
    args = ap.parse_args()

    assert args.n_test <= 3, "n-test > 3 would drop train below 11 -> bucket xs (recipe changes). Refusing."

    tar_path = args.tar or resolve_tar()
    prefix = f"tasks/{TASK_ID}/img/"

    # collect base names that have BOTH .png and .txt
    pngs, txts = {}, {}
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.startswith(prefix)]
        for m in members:
            fn = os.path.basename(m.name)
            base, ext = os.path.splitext(fn)
            if ext.lower() == ".png":
                pngs[base] = m
            elif ext.lower() == ".txt":
                txts[base] = m
        bases = sorted(set(pngs) & set(txts))
        assert len(bases) == 14, f"expected 14 pairs, found {len(bases)}"

        rng = random.Random(args.seed)
        test_bases = sorted(rng.sample(bases, args.n_test))
        train_bases = [b for b in bases if b not in test_bases]

        task_root = os.path.join(args.out_root, TASK_ID)
        train_dir = os.path.join(task_root, "img", TRAIN_SUBDIR)
        test_dir = os.path.join(task_root, "test")
        for d in (train_dir, test_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        def extract_pair(base, dst):
            for store, ext in ((pngs, ".png"), (txts, ".txt")):
                m = store[base]
                with tf.extractfile(m) as src, open(os.path.join(dst, base + ext), "wb") as out:
                    out.write(src.read())

        for b in train_bases:
            extract_pair(b, train_dir)
        for b in test_bases:
            extract_pair(b, test_dir)

    manifest = {
        "task_id": TASK_ID,
        "base_model": "John6666/nova-anime-xl-pony-v5-sdxl",
        "bucket": "s",
        "seed": args.seed,
        "n_train": len(train_bases),
        "n_test": len(test_bases),
        "train_bases": train_bases,
        "test_bases": test_bases,
        "train_dir": train_dir,
        "test_dir": test_dir,
    }
    with open(os.path.join(task_root, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] train: {len(train_bases)} pairs -> {train_dir}")
    print(f"[ok] test : {len(test_bases)} pairs -> {test_dir}")
    print(f"[ok] test bases: {test_bases}")
    print(f"[ok] manifest -> {os.path.join(task_root, 'split_manifest.json')}")


if __name__ == "__main__":
    main()
