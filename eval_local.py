#!/usr/bin/env python3
"""
eval_local.py — engine eval lokal (replikasi validator), dipakai validate_eval.py & sweep_dethrone.py.
Jalanin eval container `gradientsio/image-evaluator:basilica` di test_data LOKAL, MODELS = HF repo.
Skor = 0.25*text + 0.75*no_text. Lower better. JALANIN DI L40 (butuh docker+GPU).
"""
import json, os, subprocess, sys, tempfile, uuid

EVAL_IMAGE = "gradientsio/image-evaluator:basilica"
RESULTS_PATH = "/aplp/evaluation_results.json"
TEXT_W = 0.25


def run_eval_container(models, base_model, test_dir, model_type, gpu=0,
                       hf_cache=os.path.expanduser("~/.cache/huggingface/hub")):
    """models: list HF repo (LoRA). return {repo: {text_guided_losses, no_text_losses}} | {repo: errstr}."""
    name = f"eval-{uuid.uuid4().hex[:8]}"
    tok = os.environ.get("HF_TOKEN", "")
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
    ]
    if tok:  # akses repo LoRA private (sweep push private); public repo gak terpengaruh
        cmd += ["-e", f"HF_TOKEN={tok}", "-e", f"HUGGING_FACE_HUB_TOKEN={tok}"]
    cmd += [EVAL_IMAGE]
    safe = " ".join(cmd).replace(tok, "***") if tok else " ".join(cmd)
    print("RUN:", safe, flush=True)
    rc = subprocess.run(cmd).returncode
    out = {}
    try:
        if rc == 0:
            tmp = tempfile.mkdtemp(); dst = os.path.join(tmp, "r.json")
            subprocess.run(["docker", "cp", f"{name}:{RESULTS_PATH}", dst], check=True)
            raw = json.load(open(dst))
            for repo, r in raw.items():
                if repo == "model_params_count":
                    continue
                out[repo] = r.get("eval_loss") if isinstance(r, dict) else str(r)
        else:
            print(f"[!] eval container exit {rc}", file=sys.stderr)
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def weighted(eval_loss):
    """return (text_avg, no_text_avg, weighted). eval_loss = {text_guided_losses, no_text_losses} | float."""
    if not isinstance(eval_loss, dict):
        return float("nan"), float("nan"), float("nan")
    tg = eval_loss.get("text_guided_losses") or []
    nt = eval_loss.get("no_text_losses") or []
    tg_a = sum(tg) / len(tg) if tg else float("nan")
    nt_a = sum(nt) / len(nt) if nt else float("nan")
    return tg_a, nt_a, TEXT_W * tg_a + (1 - TEXT_W) * nt_a


def repo_for(cfg, task_key, hotkey_role):
    t = cfg["tasks"][task_key]
    hk = cfg["hotkeys"][hotkey_role]
    return f"{cfg['repo_prefix']}-{t['id']}-{hk}"
