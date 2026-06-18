#!/usr/bin/env python3
"""
eval_ab.py — bandingin recipe jalur-B vs jalur-C di held-out yang SAMA (JALANIN DI L40).
Eval deterministik validator: master_seed=42, strength 1.0, EVAL_DEFAULTS, no_text=empty-prompt.
Skor = 0.25*text_guided + 0.75*no_text. Anchor bos person = 0.086.

PRASYARAT (lo siapin di L40):
  - LoRA hasil train branch b & c udah di-push ke HF (eval container narik via HF).
    * train: git checkout jalur-b-sizeaware -> train di ds_person_v1/train -> push HF (repo B)
             git checkout jalur-c-dethrone   -> train di ds_person_v1/train -> push HF (repo C)
    * dataset SAMA (ds_person_v1/train), base model SAMA.
  - held-out test = ds_person_v1/test (5 gambar asli + .txt).
  - docker + image eval `gradientsio/image-evaluator:basilica` + GPU.

  python3 eval_ab.py --lora-b <hf_repo_b> --lora-c <hf_repo_c> \
     --base-model zenless-lab/sdxl-blue-pencil-xl-v7 \
     --test-dir ds_person_v1/test --model-type sdxl --gpu 0
"""
import argparse, json, os, subprocess, sys, tempfile, uuid

EVAL_IMAGE = "gradientsio/image-evaluator:basilica"
RESULTS_IN_CONTAINER = "/aplp/evaluation_results.json"
TEXT_W = 0.25
BOSS_ANCHOR = 0.086  # bos person final (Qwen); referensi kasar


def run_eval(models, base_model, test_dir, model_type, gpu, hf_cache):
    name = f"eval-ab-{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker", "run", "--name", name, "--gpus", f"device={gpu}", "--runtime", "nvidia",
        "-v", f"{os.path.abspath(test_dir)}:/workspace/input_data:ro",
        "-v", f"{hf_cache}:/app/validator/evaluation/ComfyUI/models/checkpoints",
        "-v", f"{hf_cache}:/app/validator/evaluation/ComfyUI/models/diffusers",
        "-e", "DATASET=/workspace/input_data",
        "-e", f"MODELS={','.join(models)}",
        "-e", f"ORIGINAL_MODEL_REPO={base_model}",
        "-e", f"MODEL_TYPE={model_type}",
        "-e", "TRANSFORMERS_ALLOW_TORCH_LOAD=true",
        EVAL_IMAGE,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    out = {}
    try:
        if rc == 0:
            tmp = tempfile.mkdtemp()
            dst = os.path.join(tmp, "results.json")
            subprocess.run(["docker", "cp", f"{name}:{RESULTS_IN_CONTAINER}", dst], check=True)
            out = json.load(open(dst))
        else:
            print(f"[!] container exit {rc}", file=sys.stderr)
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def weighted(eval_loss):
    tg = eval_loss.get("text_guided_losses") or []
    nt = eval_loss.get("no_text_losses") or []
    tg_avg = sum(tg) / len(tg) if tg else float("nan")
    nt_avg = sum(nt) / len(nt) if nt else float("nan")
    w = TEXT_W * tg_avg + (1 - TEXT_W) * nt_avg
    return tg_avg, nt_avg, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-b", required=True, help="HF repo LoRA hasil jalur-b")
    ap.add_argument("--lora-c", required=True, help="HF repo LoRA hasil jalur-c")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--test-dir", default="ds_person_v1/test")
    ap.add_argument("--model-type", default="sdxl")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface/hub"))
    args = ap.parse_args()

    res = run_eval([args.lora_b, args.lora_c], args.base_model, args.test_dir,
                   args.model_type, args.gpu, args.hf_cache)
    if not res:
        sys.exit("Eval gagal / nggak ada hasil.")

    print(f"\n{'branch':<10}{'text_guided':>13}{'no_text(75%)':>14}{'weighted':>11}{'vs_boss_0.086':>15}")
    rows = []
    for label, repo in [("jalur-b", args.lora_b), ("jalur-c", args.lora_c)]:
        r = res.get(repo)
        if not isinstance(r, dict) or "eval_loss" not in r:
            print(f"{label:<10}  (gagal: {r})"); continue
        tg, nt, w = weighted(r["eval_loss"])
        rows.append((label, w))
        print(f"{label:<10}{tg:>13.5f}{nt:>14.5f}{w:>11.5f}{w - BOSS_ANCHOR:>+15.5f}")
    if len(rows) == 2:
        b, c = rows[0][1], rows[1][1]
        delta = c - b
        print(f"\nDELTA (c - b) weighted = {delta:+.5f}  -> "
              f"{'jalur-c LEBIH BAIK' if delta < 0 else 'jalur-b lebih baik / c regресi'}")
        print("Fokus kolom no_text(75%): di situ saturasi caption_dropout paling kelihatan.")
    print(f"\nCatatan: held-out cuma {len([f for f in os.listdir(args.test_dir) if f.endswith('.png')])} gambar -> "
          "noisy. Kalau delta < ~0.003, anggap inconclusive, tambah held-out (self-gen).")


if __name__ == "__main__":
    main()
