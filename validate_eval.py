#!/usr/bin/env python3
"""
FASE 0 — validasi replikasi eval lokal. TANPA training. JALANIN DI L40.
Eval LoRA boss & challenger (dari HF) di test_data LOKAL tiap task, bandingin ke angka tabel final.
SUKSES = reproduce angka boss/chal +/- toleransi. MELESET = bug replikasi -> STOP, lapor.

Prasyarat: test_data tiap task ada lokal sbg folder (image + .txt), mis:
  data/T1/test/  data/T3/test/  data/T5/test/   (pull pakai fetch_task_dataset.py / scp dari lokal)

  python3 validate_eval.py --tasks T1 T3 T5 --data-root data --gpu 0 --tol 0.004
"""
import argparse, json, os, sys
from eval_local import run_eval_container, weighted, repo_for

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "dethrone_tasks.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T3", "T5"])
    ap.add_argument("--data-root", default="data", help="data/<TASK>/test/ = test_data lokal")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.004, help="toleransi reproduce (held-out beda -> longgar)")
    args = ap.parse_args()

    all_ok = True
    for tk in args.tasks:
        t = CFG["tasks"][tk]
        test_dir = os.path.join(args.data_root, tk, "test")
        if not os.path.isdir(test_dir):
            print(f"[{tk}] SKIP: {test_dir} nggak ada"); continue
        boss_repo, chal_repo = repo_for(CFG, tk, "boss"), repo_for(CFG, tk, "chal")
        print(f"\n===== {tk} ({t['cat']}/{t['model_type']}, base={t['base']}) =====")
        res = run_eval_container([boss_repo, chal_repo], t["base"], test_dir, t["model_type"], args.gpu)

        print(f"{'who':<6}{'weighted':>11}{'expected':>11}{'diff':>10}{'verdict':>10}")
        for role, repo, exp in [("boss", boss_repo, t.get("boss")), ("chal", chal_repo, t.get("chal"))]:
            el = res.get(repo)
            if not isinstance(el, dict):
                print(f"{role:<6}  GAGAL: {el}"); all_ok = False; continue
            _, _, w = weighted(el)
            d = w - exp if exp is not None else float("nan")
            ok = exp is not None and abs(d) <= args.tol
            all_ok = all_ok and ok
            print(f"{role:<6}{w:>11.5f}{exp if exp else 0:>11.5f}{d:>+10.5f}{'OK' if ok else 'MELESET':>10}")
        # ⚠️ held-out lokal != held-out validator -> angka bisa beda; tol longgar. Yg penting RANKING boss<chal konsisten.

    print("\n" + ("=== SEMUA OK: replikasi eval valid, lanjut FASE 1/2 ===" if all_ok else
                  "=== ADA YG MELESET: cek base/model_type/EVAL_DEFAULTS/seed, atau test_data beda dr validator. "
                  "Fokus: RANKING (boss < chal) harus konsisten walau absolut geser. ==="))
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
