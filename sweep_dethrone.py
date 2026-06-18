#!/usr/bin/env python3
"""
FASE 1 (logo SDXL +LoRA+) & FASE 2 (person) sweep. JALANIN DI L40. 1 GPU per arm (mimic tournament, JANGAN DDP).
Arm = varian recipe (BUKAN blind search). Tiap arm: train (env-override) -> push HF -> eval di test_data lokal.
Skor = 0.25*text + 0.75*no_text. Target dethrone = loss <= boss*0.97 (threshold 3%).

Alur per arm:
  1. set env DETHRONE_SWEEP_* -> panggil ./train_one.sh (lo implement utk L40-mu) -> LoRA dir
  2. push LoRA ke HF repo sementara (HF_TOKEN env)
  3. eval semua arm sekaligus (1 container) di --test-dir
Contoh:
  export HF_TOKEN=...; export HF_NS=keilrockstars
  python3 sweep_dethrone.py --phase logo --task T5 \
     --train-data-dir data/T5/train --test-dir data/T5/test --gpu 0
"""
import argparse, json, os, subprocess, sys
from eval_local import run_eval_container, weighted

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "dethrone_tasks.json")))

# conv4 DoRA base utk SDXL logo
_DORA = lambda dim, extra=(): {"network_module": "lycoris.kohya", "network_dim": dim, "network_alpha": dim,
                               "network_args": ["conv_dim=4", "conv_alpha=4", "algo=lora", "dora_wd=True", "dropout=0", *extra]}
_PLAIN = lambda dim: {"network_module": "networks.lora", "network_dim": dim, "network_alpha": dim, "network_args": []}

# ARMS: tiap arm = dict env-override. None = nggak di-set (pakai default jalur-c).
ARMS = {
    # FASE 1 — logo SDXL: flip gap 0.5% jadi >3% via LoRA+ / rank
    "logo": {
        "a_dora32":            {"NETWORK": _DORA(32)},                                 # ~challenger baseline
        "b_dora32_loraplus16": {"NETWORK": _DORA(32, ["loraplus_lr_ratio=16"])},        # +LoRA+
        "c_dora48_loraplus16": {"NETWORK": _DORA(48, ["loraplus_lr_ratio=16"])},        # +LoRA+ rank48
        "d_dora32_lp16_cd0":   {"NETWORK": _DORA(32, ["loraplus_lr_ratio=16"]), "CD": 0.0},  # logo butuh TEKS -> cd rendah (ide tambahan)
    },
    # FASE 2 Track A — person SDXL (QF blue-pencil/Ramiro). default plain-64; DoRA-64 = hipotesis
    "person-sdxl": {
        "a_plain64_cd05":  {"NETWORK": _PLAIN(64), "CD": 0.05},   # = jalur-c default
        "b_plain64_cd10":  {"NETWORK": _PLAIN(64), "CD": 0.10},   # saturasi no_text lebih
        "c_plain96_cd05":  {"NETWORK": _PLAIN(96), "CD": 0.05},   # kapasitas identitas
        "d_dora64_cd10":   {"NETWORK": _DORA(64), "CD": 0.10},    # hipotesis DoRA-64 person
    },
    # FASE 2 Track B — person Qwen (T1 David, ai-toolkit). TE-on = lever yg BOSS & CHAL belum coba
    "person-qwen": {
        "a_teoff_cd05": {"QWEN_TE": "false", "CD": 0.05},         # = jalur-c default = boss
        "b_teon_cd05":  {"QWEN_TE": "true",  "CD": 0.05},         # EDGE candidate
        "c_teon_cd10":  {"QWEN_TE": "true",  "CD": 0.10},
    },
}


def train_arm(arm_env, lora_out, args):
    env = dict(os.environ)
    if arm_env.get("NETWORK") is not None:
        env["DETHRONE_SWEEP_NETWORK"] = json.dumps(arm_env["NETWORK"])
    for k, ek in [("CD", "DETHRONE_SWEEP_CD"), ("STEPS", "DETHRONE_SWEEP_STEPS"),
                  ("QWEN_TE", "DETHRONE_SWEEP_QWEN_TE"), ("LINEAR", "DETHRONE_SWEEP_LINEAR")]:
        if arm_env.get(k) is not None:
            env[ek] = str(arm_env[k])
    # ./train_one.sh <model_type> <base> <train_data_dir> <trigger> <out_dir> <gpu>
    cmd = ["./train_one.sh", args.model_type, args.base_model, args.train_data_dir,
           args.trigger or "", lora_out, str(args.gpu)]
    print("TRAIN:", " ".join(cmd), "| env:", {k: env[k] for k in env if k.startswith("DETHRONE_")}, flush=True)
    subprocess.run(cmd, env=env, check=True)


def push_hf(lora_dir, repo):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, exist_ok=True, private=True)
    api.upload_folder(folder_path=lora_dir, repo_id=repo)
    return repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(ARMS))
    ap.add_argument("--task", help="ambil base/model_type/boss dari dethrone_tasks.json (mis T5/QF/T1)")
    ap.add_argument("--train-data-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--base-model"); ap.add_argument("--model-type"); ap.add_argument("--trigger")
    ap.add_argument("--boss-loss", type=float)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--hf-ns", default=os.environ.get("HF_NS"))
    ap.add_argument("--skip-train", action="store_true", help="LoRA udah ke-push, eval doang")
    args = ap.parse_args()

    if args.task:
        t = CFG["tasks"][args.task]
        args.base_model = args.base_model or t["base"]
        args.model_type = args.model_type or t["model_type"]
        args.trigger = args.trigger if args.trigger is not None else t.get("trigger")
        args.boss_loss = args.boss_loss or t.get("boss")
    assert args.base_model and args.model_type and args.hf_ns, "butuh --base-model --model-type --hf-ns (atau --task + HF_NS)"
    target = args.boss_loss * 0.97 if args.boss_loss else None

    arms = ARMS[args.phase]
    repos = {}
    for name, arm_env in arms.items():
        repo = f"{args.hf_ns}/sweep-{args.phase}-{name}"
        repos[name] = repo
        if args.skip_train:
            continue
        out = os.path.abspath(f"sweep_out/{args.phase}/{name}")
        os.makedirs(out, exist_ok=True)
        train_arm(arm_env, out, args)
        push_hf(out, repo)

    print("\nEval semua arm (1 container)...")
    res = run_eval_container(list(repos.values()), args.base_model, args.test_dir, args.model_type, args.gpu)

    rows = []
    print(f"\n{'arm':<22}{'text':>10}{'no_text':>10}{'weighted':>11}{'vs_boss':>10}")
    for name, repo in repos.items():
        el = res.get(repo)
        if not isinstance(el, dict):
            print(f"{name:<22}  GAGAL: {el}"); continue
        tg, nt, w = weighted(el)
        rows.append((name, w))
        vs = (w - args.boss_loss) if args.boss_loss else float("nan")
        flag = ""
        if target is not None:
            flag = " <= TARGET (dethrone!)" if w <= target else (" beat boss (tipis, <3%)" if w < args.boss_loss else "")
        print(f"{name:<22}{tg:>10.5f}{nt:>10.5f}{w:>11.5f}{vs:>+10.5f}{flag}")
    if args.boss_loss:
        print(f"\nboss={args.boss_loss:.5f}  target(<=3%)={target:.5f}. Arm dgn weighted <= target = flip task.")
    if rows:
        best = min(rows, key=lambda r: r[1])
        print(f"BEST arm: {best[0]} ({best[1]:.5f}). Fokus DELTA no_text antar-arm. "
              f"Test {len([f for f in os.listdir(args.test_dir) if f.endswith('.png')])} img -> kalau menang tipis, RE-VALIDASI di task logo lain (T6).")


if __name__ == "__main__":
    main()
