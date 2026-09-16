#!/usr/bin/env python
"""Seed means for run_sitl_nm.sh, set against the motor-era sitl_agg single run.

The reference numbers are read from the files that published them
(stage4_openloop_agg.json, observability_agg_model.json), not retyped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
METRICS = ("ATE", "TDE", "ekf_ATE", "ekf_TDE", "cosine", "magnitude_ratio")


def load_run(d: Path, tag: str) -> dict:
    ol = json.loads((d / f"openloop_{tag}.json").read_text())["summary"]
    dr = json.loads((d / f"direction_{tag}.json").read_text())["flight_6.npz"]["direction"]
    return {"ATE": ol["gt"]["ATE"], "TDE": ol["gt"]["TDE"],
            "ekf_ATE": ol["ekf"]["ATE"], "ekf_TDE": ol["ekf"]["TDE"],
            "cosine": dr["cosine"], "magnitude_ratio": dr["magnitude_ratio"]}


def reference() -> dict:
    ol = json.loads((RESULTS / "stage4_openloop_agg.json").read_text())["summary"]
    dr = json.loads((RESULTS / "observability_agg_model.json").read_text())["flight_6.npz"]["direction"]
    return {"ATE": ol["gt"]["ATE"], "TDE": ol["gt"]["TDE"],
            "ekf_ATE": ol["ekf"]["ATE"], "ekf_TDE": ol["ekf"]["TDE"],
            "cosine": dr["cosine"], "magnitude_ratio": dr["magnitude_ratio"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RESULTS / "sitl_nm")
    ap.add_argument("--seeds", nargs="+", default=["42", "1", "2", "3"])
    args = ap.parse_args()

    out = {"reference_sitl_agg_motor_single_run": reference()}
    for kind in ("zeroshot", "finetuned"):
        per_seed = {s: load_run(args.dir, f"{kind}_s{s}") for s in args.seeds}
        out[kind] = {
            "per_seed": per_seed,
            "mean": {m: float(np.mean([r[m] for r in per_seed.values()])) for m in METRICS},
            "std": {m: float(np.std([r[m] for r in per_seed.values()], ddof=1))
                    if len(per_seed) > 1 else 0.0 for m in METRICS},
        }
        meta = {s: json.loads(p.read_text()) for s in args.seeds
                if (p := args.dir.parents[2] / "runs" / f"sitl_agg_nm_s{s}" / "finetune_meta.json").exists()}
        if kind == "finetuned" and meta:
            out[kind]["best_test_rmse"] = {s: m["best_test_rmse"] for s, m in meta.items()}

    out["note"] = ("held-out flight_6, 284.8 s. best.pt is chosen by window RMSE on this same "
                   "flight (mario.train.train), as it was for sitl_agg, so the finetuned numbers "
                   "carry checkpoint-selection optimism; the reference carries the same.")
    (args.dir / "summary.json").write_text(json.dumps(out, indent=2))

    ref = out["reference_sitl_agg_motor_single_run"]
    print(f"\n{'':<24}{'ATE [m]':>14}{'TDE [%]':>14}{'EKF-att ATE':>14}{'cos':>14}")
    print(f"{'sitl_agg (motor, 1 run)':<24}{ref['ATE']:>14.3f}{ref['TDE']:>14.2f}"
          f"{ref['ekf_ATE']:>14.3f}{ref['cosine']:>+14.3f}")
    for kind in ("zeroshot", "finetuned"):
        m, sd = out[kind]["mean"], out[kind]["std"]
        cells = [f"{m[k]:.3f}±{sd[k]:.3f}" if k != "TDE" else f"{m[k]:.2f}±{sd[k]:.2f}"
                 for k in ("ATE", "TDE", "ekf_ATE", "cosine")]
        print(f"{'NM ' + kind + ' (4 seeds)':<24}" + "".join(f"{c:>14}" for c in cells))
    print(f"\nwrote {args.dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
