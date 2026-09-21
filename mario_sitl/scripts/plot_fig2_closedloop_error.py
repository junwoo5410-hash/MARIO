#!/usr/bin/env python
"""Fig. 2 (HuTech abstract §3.3) -- closed-loop position error over time.

Position error at each mission sample is ||(est - est_origin) - (gt - gt_origin)||, the
definition continuous_mission_node.py uses for its summary, so the two continuous curves
end exactly on the published est_error_final_m. Ground truth is
/fmu/out/vehicle_local_position_groundtruth, logged by the mission nodes.

  Station keeping           closedloop_run_mario.json   t = 0 at the takeoff event (origin latch)
  Continuous (3 laps)       continuous_run.json         t = 0 at the GPS cut
  Continuous (yaw-aligned)  continuous_run_yf.json      t = 0 at the GPS cut

Every logged sample in the MARIO-only span is plotted as recorded: no smoothing, no
resampling, no truncation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"


def _valid(samples):
    return [x for x in samples
            if not (np.isnan(np.asarray(x["gt"], float)).any()
                    or np.isnan(np.asarray(x["est"], float)).any())]


def _error(rows, est_origin, gt_origin, t0):
    t = np.array([x["t"] for x in rows]) - t0
    est = np.array([x["est"] for x in rows], float) - np.asarray(est_origin, float)
    gt = np.array([x["gt"] for x in rows], float) - np.asarray(gt_origin, float)
    return t, np.linalg.norm(est - gt, axis=1), gt


def continuous(name: str):
    d = json.loads((R / name).read_text())
    s = d["summary"]
    rows = [x for x in _valid(d["samples"]) if x["phase"] == "lap_mario"]
    t, err, gt = _error(rows, s["ekf_origin"], s["gt_origin"], rows[0]["t"])
    dist = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    return t, err, {"final": float(err[-1]), "published_final": s["est_error_final_m"],
                    "distance": dist, "published_distance": s["distance_flown_m"]}


def station_keeping():
    d = json.loads((R / "closedloop_run_mario.json").read_text())
    s = d["summary"]
    t0 = next(e["t"] for e in d["events"] if e["phase"] == "takeoff")
    rows = [x for x in _valid(d["samples"]) if x["t"] >= t0]
    t, err, gt = _error(rows, s["ekf_origin"], s["gt_origin"], t0)
    dist = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    return t, err, {"final": float(err[-1]), "distance": dist,
                    "displacement": float(np.linalg.norm(gt[-1]))}


def main() -> int:
    sk_t, sk_e, sk = station_keeping()
    c3_t, c3_e, c3 = continuous("continuous_run.json")
    yf_t, yf_e, yf = continuous("continuous_run_yf.json")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, ax = plt.subplots(figsize=(4.4, 1.33))
    fig.patch.set_facecolor("white")
    ax.plot(sk_t, sk_e, color="#c0392b", lw=1.6, ls="-", label="Station keeping")
    ax.plot(c3_t, c3_e, color="#1f5fa8", lw=1.0, ls="-", label="Continuous (3 laps)")
    ax.plot(yf_t, yf_e, color="#4a90d9", lw=1.0, ls="--", label="Continuous (yaw-aligned)")
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("Position error [m]", fontsize=8)
    ax.tick_params(labelsize=7, length=2.5, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=2.2,
              borderaxespad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.2)

    out = R / "figures" / "fig2_closedloop_error.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")

    report = {
        "station_keeping": {**sk, "t_end": float(sk_t[-1]), "samples": len(sk_t),
                            "monotone_fraction": float(np.mean(np.diff(sk_e) >= 0))},
        "continuous_3laps": {**c3, "t_end": float(c3_t[-1]), "samples": len(c3_t)},
        "continuous_yaw_aligned": {**yf, "t_end": float(yf_t[-1]), "samples": len(yf_t)},
    }
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
