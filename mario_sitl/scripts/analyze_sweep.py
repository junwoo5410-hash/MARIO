#!/usr/bin/env python
"""Stage 6 — score sweep runs that never reached the phase the mission node scores.

The mission node reports arrival error and hover drift only if the run reaches its hover
phase. Under MARIO-only closed loop it frequently does not: the estimate drifts faster than
the controller can settle inside the 0.3 m arrival radius, so takeoff times out and every
summary field comes back null. That is a result, not a gap -- but it has to be measured
from the sample trace instead of the summary.

Per run this reports how far the airframe actually went while being told to climb and hold,
which is the quantity that matters when the loop is closed on a bad estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results"


def score(path: Path) -> dict | None:
    d = json.loads(path.read_text())
    s, sm = d["summary"], d.get("samples", [])
    rows = [x for x in sm if not (np.isnan(np.asarray(x["gt"], dtype=float)).any()
                                  or np.isnan(np.asarray(x["est"], dtype=float)).any())]
    if len(rows) < 50:
        return None

    t = np.array([x["t"] for x in rows]); t -= t[0]
    gt = np.array([x["gt"] for x in rows], dtype=float)
    est = np.array([x["est"] for x in rows], dtype=float)
    sp = np.array([x["setpoint"] for x in rows], dtype=float)
    phases = [x["phase"] for x in rows]

    gt_o = np.array(s.get("gt_origin") or gt[0], dtype=float)
    ekf_o = np.array(s.get("ekf_origin") or est[0], dtype=float)
    gt_d, est_d, sp_d = gt - gt_o, est - ekf_o, sp - ekf_o

    err = np.linalg.norm(est_d - gt_d, axis=1)
    excursion = np.linalg.norm(gt_d - sp_d, axis=1)
    horiz = np.linalg.norm((gt_d - sp_d)[:, :2], axis=1)

    reached = "hover" in phases
    return {
        "reached_hover": reached,
        "duration_s": float(t[-1]),
        "gt_excursion_max_m": float(excursion.max()),
        "gt_excursion_final_m": float(excursion[-1]),
        "gt_horizontal_excursion_max_m": float(horiz.max()),
        "est_error_max_m": float(err.max()),
        "est_error_final_m": float(err.max()),
        "est_error_rate_m_per_s": float(np.polyfit(t, err, 1)[0]) if len(t) > 2 else 0.0,
        "est_offset_at_latch_m": float(np.linalg.norm(ekf_o - gt_o)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=R / "sweep")
    ap.add_argument("--out", type=Path, default=R / "stage6_sweep_analysis.json")
    args = ap.parse_args()

    files = sorted(f for f in args.dir.glob("*.json") if "_shadow" not in f.name)
    if not files:
        print(f"no runs under {args.dir}")
        return 1

    rows = {}
    print(f"{'run':<28}{'hover?':>7}{'excur max':>11}{'horiz':>8}"
          f"{'est err':>9}{'rate':>8}{'latch':>7}")
    print("-" * 80)
    for f in files:
        r = score(f)
        if not r:
            continue
        rows[f.stem] = r
        print(f"{f.stem:<28}{str(r['reached_hover']):>7}"
              f"{r['gt_excursion_max_m']:>11.2f}{r['gt_horizontal_excursion_max_m']:>8.2f}"
              f"{r['est_error_max_m']:>9.2f}{r['est_error_rate_m_per_s']:>8.2f}"
              f"{r['est_offset_at_latch_m']:>7.2f}")

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    print("\nGPS baseline for comparison: arrival 0.126 m, settled deviation 0.59 m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
