#!/usr/bin/env python
"""Collect the transfer runs into a seed-averaged table.

Single-seed numbers are not reported as results. The Blackbird work established that
this pipeline is chaotic enough that a 0.011 % learning-rate change moves seen ATE by
14 %, and that every apparent single-seed win there failed to replicate. So the mean
over seeds is the unit of claim, and the per-seed spread is printed beside it so a
difference smaller than the spread is visible as such.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "transfer"

ORDER = ["scratch", "head", "trunk", "full"]


def main() -> int:
    want = sys.argv[1] if len(sys.argv) > 1 else None
    runs = Path(sys.argv[2]) if len(sys.argv) > 2 else RUNS
    by = defaultdict(list)
    for path in sorted(runs.glob("*/results.json")):
        r = json.loads(path.read_text())
        if want and r["dataset"] != want:
            continue
        by[(r["dataset"], r.get("select", "rmse"), r["mode"], r.get("n_train_seqs"),
            round(r["lr"], 9))].append(r)

    if not by:
        print("no runs found")
        return 1

    for ds in sorted({k[0] for k in by}):
        print(f"\n=== {ds} ===")
        print(f"{'select':<8}{'mode':<9}{'lr':>10}{'n_tr':>5}{'seeds':>7}{'trainable':>11}"
              f"{'ATE [m]':>10}{'+-':>8}{'TDE [%]':>10}{'+-':>8}")
        print("-" * 86)
        keys = [k for k in by if k[0] == ds]
        keys.sort(key=lambda k: (k[1], -k[4], ORDER.index(k[2]) if k[2] in ORDER else 9, k[3] or 0))
        for k in keys:
            rs = by[k]
            ate = np.array([r["result"]["mean_ate"] for r in rs])
            tde = np.array([r["result"]["mean_tde"] for r in rs])
            frac = rs[0]["trainable_params"] / rs[0]["total_params"] * 100
            print(f"{k[1]:<8}{k[2]:<9}{k[4]:>10.3g}{k[3] or '-':>5}{len(rs):>7}{frac:>10.1f}%"
                  f"{ate.mean():>10.3f}{ate.std():>8.3f}{tde.mean():>10.2f}{tde.std():>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
