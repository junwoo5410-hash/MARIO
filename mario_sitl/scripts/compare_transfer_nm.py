#!/usr/bin/env python
"""NM vs motor-era EuRoC transfer, full-trajectory basis, against the stationary baseline.

Reads the two full_rollout_rescore.json files (motor-era runs/transfer, no-motor
runs/transfer_nm) and prints one table per data budget. Only the runs that share the
published protocol are compared: dataset euroc, select=ate, the default per-mode lr, and
our2 dropped. The stationary baseline -- the ATE of answering "the drone never moved" --
is printed alongside, because on EuRoC every arm has so far sat below it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "mario_sitl" / "results"
SOURCE_LR = 2.5757267464272164e-4          # configs/trial8.yaml
DEFAULT_LR = {"scratch": SOURCE_LR}        # every other mode: a tenth of it
ORDER = ["zeroshot", "scratch", "head", "trunk", "full"]
MINUTES = {1: 0.91, 2: 2.24, 4: 5.75, 6: 8.60}   # FINETUNE_METHOD.md §2.7, windowed span


def load(path: Path) -> dict:
    rows = [r for r in json.loads(path.read_text()) if r["dataset"] == "euroc"]
    keep = defaultdict(list)
    for r in rows:
        if r["mode"] == "zeroshot":
            keep[("zeroshot", 0)].append(r)
            continue
        if r.get("select") != "ate":
            continue
        want = DEFAULT_LR.get(r["mode"], SOURCE_LR / 10.0)
        if abs(r["lr"] - want) > 1e-9:
            continue
        keep[(r["mode"], r["n_train"])].append(r)
    return keep


def cell(rows: list) -> str:
    if not rows:
        return "-"
    a = np.array([r["full_ate"] for r in rows])
    return f"{a.mean():6.2f} ±{a.std():4.2f} ({len(a)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nm", type=Path, default=RESULTS / "transfer_nm" / "full_rollout_rescore.json")
    ap.add_argument("--motor", type=Path, default=RESULTS / "transfer" / "full_rollout_rescore.json")
    ap.add_argument("--out", type=Path, default=RESULTS / "transfer_nm" / "nm_vs_motor.json")
    args = ap.parse_args()

    nm, m = load(args.nm), load(args.motor)
    base = json.loads((RESULTS / "stationary_baseline.json").read_text())["euroc_test"]
    base_ate = base["mean_ate"] if "mean_ate" in base else np.mean(
        [v["ATE"] for v in base["per_trajectory"].values()])

    print(f"EuRoC test, full-trajectory ATE [m], mean ±std (seeds). "
          f"Stationary baseline {base_ate:.2f} m — below it means worse than 'never moved'.\n")
    print(f"{'data':<22}{'mode':<9}{'no-motor (NM)':>22}{'motor-era (M)':>22}")
    out = {}
    for n in (0, 1, 2, 4, 6):
        label = "zero-shot" if n == 0 else f"{n} seq ({MINUTES[n]:.2f} min)"
        for mode in ORDER:
            k = (mode, n)
            if not (nm.get(k) or m.get(k)):
                continue
            print(f"{label:<22}{mode:<9}{cell(nm.get(k, [])):>22}{cell(m.get(k, [])):>22}")
            label = ""
            out[f"{mode}_n{n}"] = {
                "nm_full_ate": [r["full_ate"] for r in nm.get(k, [])],
                "motor_full_ate": [r["full_ate"] for r in m.get(k, [])],
            }
        print()

    best = {}
    for n in (1, 2, 4, 6):
        arms = {mode: np.mean([r["full_ate"] for r in nm[(mode, n)]])
                for mode in ORDER if nm.get((mode, n))}
        if arms:
            b = min(arms, key=arms.get)
            best[n] = (b, arms[b], arms[b] < base_ate)
            print(f"n={n}: best NM arm = {b} ({arms[b]:.2f} m), "
                  f"{'beats' if arms[b] < base_ate else 'BELOW'} the stationary baseline")

    out["stationary_baseline_ate"] = float(base_ate)
    out["best_nm_arm_per_budget"] = {str(k): {"arm": v[0], "ate": float(v[1]),
                                              "beats_baseline": bool(v[2])}
                                     for k, v in best.items()}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
