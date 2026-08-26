#!/usr/bin/env python
"""Stage 5 — score a closed-loop run and separate the two failure modes.

The spec asks for a distinction that a single error number cannot express:

  * the estimate is wrong but the airframe is still  -> the controller has not acted on
    the error yet; an early symptom.
  * the airframe actually moved by the estimate error -> the controller believed a wrong
    estimate and "corrected" reality to match it. This is the closed-loop-specific
    failure, and open-loop evaluation cannot reveal it.

Both are measured from the same samples. The controller drives the ESTIMATE onto the
setpoint, so with estimate error ``err = est_disp - gt_disp`` the airframe lands at
``gt_disp ~= sp_disp - err``. Therefore ``deviation = gt_disp - sp_disp`` should equal
``-err`` exactly when the controller is chasing the error. The projection of deviation
onto -err is reported as ``chase_ratio``: ~1 means reality was bent to match the estimate,
~0 means the estimate drifted while the airframe stayed put.

Displacements are taken from each frame's own origin. EKF2's local origin is offset from
the simulator origin by the filter's own error, so comparing raw positions would charge
the run for a frame offset it never committed (the same mistake Stage 1 made at first).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    s = [x for x in d["samples"]
         if not (np.isnan(x["gt"][0]) or np.isnan(x["est"][0]))]
    if not s:
        raise SystemExit(f"{path}: no samples with both GT and estimate")
    return {"summary": d["summary"], "events": d["events"], "samples": s}


def analyse(run: dict) -> dict:
    s = run["samples"]
    t = np.array([x["t"] for x in s])
    t -= t[0]
    sp = np.array([x["setpoint"] for x in s])
    est = np.array([x["est"] for x in s])
    gt = np.array([x["gt"] for x in s])
    phase = [x["phase"] for x in s]

    # Each series measured from its own origin (see module docstring). The mission node
    # records both origins at the same instant; fall back to the first sample only if an
    # older run lacks them. Subtracting sp[0] instead would be wrong: sample 0's setpoint
    # is already the takeoff target, so that cancels the commanded displacement itself and
    # reports a climb as deviation.
    summ = run.get("summary", {})
    ekf_o = np.array(summ.get("ekf_origin") or est[0], dtype=float)
    gt_o = np.array(summ.get("gt_origin") or gt[0], dtype=float)
    est_d, gt_d, sp_d = est - ekf_o, gt - gt_o, sp - ekf_o

    err = est_d - gt_d                       # what the filter got wrong
    dev = gt_d - sp_d                        # where the airframe actually ended up
    err_n = np.linalg.norm(err, axis=1)
    dev_n = np.linalg.norm(dev, axis=1)

    # The chase test is only meaningful while the airframe is meant to be holding station.
    # During cruise the setpoint is stepped to the far waypoint, so a large deviation is
    # ordinary tracking lag, not the controller believing a bad estimate -- scoring the
    # whole run rated the passing GPS baseline as a 5.2 m failure.
    settled = np.array([p == "hover" for p in phase])
    if not settled.any():
        settled = t >= t[-1] * 0.5

    denom = (err * err).sum(1)
    chase = np.where(denom > 1e-9, (dev * -err).sum(1) / np.maximum(denom, 1e-9), 0.0)
    strong = settled & (err_n > 0.1)          # only meaningful once the error is real

    # Divergence: is the estimate error growing rather than settling?
    half = len(err_n) // 2
    slope = float(np.polyfit(t, err_n, 1)[0]) if len(t) > 2 else 0.0

    out = {
        "duration_s": float(t[-1]),
        "est_error_final_m": float(err_n[-1]),
        "est_error_max_m": float(err_n.max()),
        "est_error_mean_m": float(err_n.mean()),
        "est_error_slope_m_per_s": slope,
        "est_error_first_half_mean_m": float(err_n[:half].mean()),
        "est_error_second_half_mean_m": float(err_n[half:].mean()),
        "diverging": bool(slope > 0.01 and err_n[half:].mean() > err_n[:half].mean()),
        "vehicle_deviation_final_m": float(dev_n[-1]),
        "vehicle_deviation_settled_max_m": float(dev_n[settled].max()) if settled.any() else 0.0,
        "vehicle_deviation_transient_max_m": float(dev_n.max()),  # includes cruise lag
        "chase_ratio": float(np.median(chase[strong])) if strong.any() else 0.0,
        "chase_samples": int(strong.sum()),
    }

    hover = [i for i, p in enumerate(phase) if p == "hover"]
    if hover:
        h = gt[hover]
        out["hover_drift_m"] = float(np.linalg.norm(h[-1] - h[0]))
        out["hover_excursion_m"] = float(np.linalg.norm(h - h[0], axis=1).max())
        out["hover_seconds"] = float(t[hover[-1]] - t[hover[0]])

    # A high chase_ratio alone is not a failure: the GPS baseline also follows its filter,
    # just by a small amount. Magnitude while settled is what separates the two.
    settled_dev = out["vehicle_deviation_settled_max_m"]
    out["interpretation"] = (
        f"controller bent reality to match a wrong estimate "
        f"({settled_dev:.2f} m while holding station)"
        if out["chase_ratio"] > 0.5 and settled_dev > 1.0 else
        f"controller follows the filter, but only by {settled_dev:.2f} m while settled"
        if out["chase_ratio"] > 0.5 else
        "estimate drifted, airframe held station"
        if out["est_error_max_m"] > 0.3 else
        "estimate tracked ground truth")
    return {**out, "_t": t, "_err": err_n, "_dev": dev_n, "_gt": gt_d, "_est": est_d,
            "_sp": sp_d, "_phase": phase}


def plot(a: dict, out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(13, 4.6))

    ax = fig.add_subplot(1, 3, 1)
    ax.plot(a["_t"], a["_err"], color="#c1443c", lw=1.3, label="estimate error |est-gt|")
    ax.plot(a["_t"], a["_dev"], color="#3b6ea5", lw=1.3, label="airframe deviation |gt-sp|")
    for ph in ("hover", "land"):
        idx = [i for i, p in enumerate(a["_phase"]) if p == ph]
        if idx:
            ax.axvline(a["_t"][idx[0]], color="#999", ls=":", lw=1)
            ax.text(a["_t"][idx[0]], ax.get_ylim()[1] * 0.95, ph, fontsize=7, color="#666")
    ax.set_xlabel("time [s]"); ax.set_ylabel("[m]")
    ax.set_title("error over time", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot(*a["_gt"].T, color="#333", lw=1.4, label="ground truth")
    ax.plot(*a["_est"].T, color="#c1443c", lw=1.2, label="EKF2 estimate")
    ax.plot(*a["_sp"].T, color="#1a7a3e", lw=1.0, ls="--", label="setpoint")
    ax.set_xlabel("N [m]"); ax.set_ylabel("E [m]"); ax.set_zlabel("D [m]")
    ax.set_title("trajectories (NED, from origin)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(a["_gt"][:, 0], a["_gt"][:, 1], color="#333", lw=1.4, label="ground truth")
    ax.plot(a["_est"][:, 0], a["_est"][:, 1], color="#c1443c", lw=1.2, label="EKF2")
    ax.plot(a["_sp"][:, 0], a["_sp"][:, 1], color="#1a7a3e", lw=1.0, ls="--", label="setpoint")
    ax.scatter(0, 0, color="#1a7a3e", s=25)
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]"); ax.set_aspect("equal", "datalim")
    ax.set_title("horizontal track", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    for a_ in fig.axes:
        if hasattr(a_, "spines") and a_.name != "3d":
            a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RESULTS / "closedloop_run_mario.json")
    ap.add_argument("--label", default="MARIO closed loop (no GPS)")
    args = ap.parse_args()

    run = load(args.run)
    a = analyse(run)
    report = {k: v for k, v in a.items() if not k.startswith("_")}

    print(f"\n{args.label}  ({args.run.name})")
    print("-" * 62)
    for k, v in report.items():
        print(f"  {k:<32} {v if not isinstance(v, float) else round(v, 4)}")

    plot(a, RESULTS / "figures" / f"{args.run.stem}.png", args.label)
    out = args.run.with_name(args.run.stem + "_analysis.json")
    out.write_text(json.dumps({"mission_summary": run["summary"], "analysis": report}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
