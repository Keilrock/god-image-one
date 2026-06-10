#!/usr/bin/env python3
"""
Parse the validator's /aplp/evaluation_results.json into an epoch -> score table.

The scorer (validator/evaluation/eval_diffusion.py) outputs per repo:
  results[repo_id] = {"eval_loss": {"text_guided_losses": [...], "no_text_losses": [...]}}
where each list has one img2img L2 loss per test image (lower = better).

We report mean text-guided loss, mean no-text loss, and their mean, per epoch, and
flag the minimum (the overfit "knee" = peak quality before loss climbs again).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="evaluation_results.json from the eval container")
    ap.add_argument("--repos", required=True, help="repos.json (repo_id -> epoch) from upload step")
    args = ap.parse_args()

    results = json.load(open(args.results))
    repo_epoch = json.load(open(args.repos))

    rows = []
    for repo_id, ep in repo_epoch.items():
        entry = results.get(repo_id)
        if not isinstance(entry, dict) or "eval_loss" not in entry:
            rows.append((ep, repo_id, None, None, None, f"MISSING/ERROR: {entry}"))
            continue
        el = entry["eval_loss"]
        tg = float(np.mean(el["text_guided_losses"])) if el.get("text_guided_losses") else float("nan")
        nt = float(np.mean(el["no_text_losses"])) if el.get("no_text_losses") else float("nan")
        rows.append((ep, repo_id, tg, nt, (tg + nt) / 2, ""))

    rows.sort(key=lambda r: r[0])
    print(f"\n{'epoch':>5} {'text_guided':>12} {'no_text':>10} {'mean_loss':>10}   note")
    print("-" * 60)
    best = min((r for r in rows if r[4] is not None), key=lambda r: r[4], default=None)
    for ep, repo, tg, nt, mean, note in rows:
        star = "  <= MIN (knee)" if best and repo == best[1] else ""
        if tg is None:
            print(f"{ep:>5} {'--':>12} {'--':>10} {'--':>10}   {note}")
        else:
            print(f"{ep:>5} {tg:>12.5f} {nt:>10.5f} {mean:>10.5f}{star}")
    if best:
        print(f"\nLowest mean loss at epoch {best[0]} (peak quality before overfit).")
        print("Interpretation: loss falling = still learning; loss rising after the min = overfitting.")
    print("\nNOTE: absolute values are NOT the tournament leaderboard score (different held-out")
    print("images + trained on 11 not the full split). The SHAPE / knee epoch is the valid signal.")


if __name__ == "__main__":
    main()
