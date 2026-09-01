#!/usr/bin/env python
"""Stage 6 — is the cov head's uncertainty worth anything?

The network emits a per-axis variance alongside each displacement, trained with a Gaussian
NLL (residual^2 / cov + log cov). Stage 5 maps it into VehicleOdometry.velocity_variance,
so EKF2 weights MARIO's velocity by it. That is only sound if the number tracks the actual
error. This checks three things:

  calibration  -- does predicted sigma match observed |error|? A perfectly calibrated
                  estimator has z = error/sigma with unit variance.
  ranking      -- does higher predicted sigma pick out the samples that are actually worse?
                  Spearman correlation, which is what matters for a filter deciding how much
                  to trust a measurement.
  sharpness    -- is sigma nearly constant? A head that always reports the same number is
                  perfectly useless even if its average magnitude is right.

Ground truth here is Gazebo's velocity, so this measures MARIO's error directly rather
than EKF2's post-fusion state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1] / "results"
STEP_DT = 0.09


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[str(R / "stage5b_shadow.json"),
                                                  str(R / "stage2_shadow.json")])
    ap.add_argument("--out", type=Path, default=R / "stage6_uncertainty.json")
    args = ap.parse_args()

    report = {}
    for path in args.runs:
        p = Path(path)
        if not p.exists():
            continue
        rec = json.loads(p.read_text())["records"]
        rows = [r for r in rec
                if not np.isnan(np.asarray(r["gt_vel_ned"], dtype=float)).any()]
        if len(rows) < 200:
            continue

        vel = np.array([r["vel_ned"] for r in rows])
        gt = np.array([r["gt_vel_ned"] for r in rows], dtype=float)
        # cov is a variance on the 0.09 s displacement step; convert to a velocity sigma
        cov = np.array([r["cov"] for r in rows])
        sigma = np.sqrt(np.maximum(cov, 1e-12)) / STEP_DT

        err = np.abs(vel - gt)
        z = err / np.maximum(sigma, 1e-9)

        per_axis = {}
        for i, ax in enumerate("NED"):
            rho = spearmanr(sigma[:, i], err[:, i]).statistic
            per_axis[ax] = {
                "sigma_mean": float(sigma[:, i].mean()),
                "error_mean": float(err[:, i].mean()),
                "ratio_error_over_sigma": float(err[:, i].mean() / max(sigma[:, i].mean(), 1e-9)),
                "spearman_sigma_vs_error": float(rho),
                "sigma_cv": float(sigma[:, i].std() / max(sigma[:, i].mean(), 1e-9)),
                "z_std": float(z[:, i].std()),
            }

        rho_all = spearmanr(sigma.ravel(), err.ravel()).statistic
        report[p.name] = {
            "samples": len(rows),
            "per_axis": per_axis,
            "spearman_overall": float(rho_all),
            # Sharpness: a head reporting a near-constant sigma carries no information,
            # however well its average happens to match.
            "sigma_cv_overall": float(sigma.std() / max(sigma.mean(), 1e-9)),
            "z_std_overall": float(z.std()),
        }

        print(f"\n{p.name}  (n={len(rows)})")
        print(f"{'axis':>5}{'sigma':>10}{'|error|':>10}{'err/sigma':>11}"
              f"{'spearman':>10}{'sigma CV':>10}")
        for ax, v in per_axis.items():
            print(f"{ax:>5}{v['sigma_mean']:>10.3f}{v['error_mean']:>10.3f}"
                  f"{v['ratio_error_over_sigma']:>11.2f}"
                  f"{v['spearman_sigma_vs_error']:>+10.3f}{v['sigma_cv']:>10.3f}")
        print(f"  overall spearman {rho_all:+.3f} | sigma CV "
              f"{report[p.name]['sigma_cv_overall']:.3f} | z std "
              f"{report[p.name]['z_std_overall']:.2f} (1.0 = calibrated)")

    if report:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
