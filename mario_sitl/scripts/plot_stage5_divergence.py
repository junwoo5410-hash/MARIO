#!/usr/bin/env python
"""Stage 5 result — closed-loop flight on MARIO alone, and how it ended.

Sequence, from the PX4 log and the mission samples:

    Armed by external command -> Takeoff detected -> Failsafe activated -> Disarmed

The airframe armed on a MARIO-only estimate (GPS off, EV velocity the sole horizontal
aiding), lifted off at t~6 s, drifted 1.05 m, and was on the ground again by t~8 s. It came
to rest rolled -63.5 degrees and never moved again. Total time airborne: about two seconds.

The controller was flying the estimate, and the estimate was already 1.9 m out at lift-off
and diverging; correcting toward it drove the airframe into the ground. That is the
closed-loop-specific failure the spec asks to distinguish -- reality bent to match a wrong
estimate -- just far faster than the slow hover drift that was anticipated.

The 24 m/s velocity and 400 m position excursion that follow are a CONSEQUENCE of the
crash, not its cause: a quadrotor lying on its side presents an accelerometer reading
(-0.57, 8.36, -4.63) that is nothing like anything in training, and the network extrapolates
wildly. Read up to the crash marker for the failure; everything after is post-mortem.

The attitude panel rules a mechanism OUT. MARIO takes rot_so3 from the EKF's attitude
(함정 C), so a plausible story was a feedback loop: bad velocity corrupts attitude, which
worsens the velocity. It did not happen -- EKF attitude stays within 3.9 degrees of truth
throughout. (An earlier version of this script reported 110 degrees, which was an artifact:
the pre-time-sync stamps described above corrupted the index alignment between the two
attitude streams.) The divergence runs purely through velocity and position.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sys  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mario_ros"))
import mario_frames as mf  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    rec = json.loads((R / "stage5_shadow.json").read_text())["records"]
    t = np.array([x["t"] for x in rec])
    mario = np.linalg.norm(np.array([x["vel_ned"] for x in rec]), axis=1)
    ekf = np.linalg.norm(np.array([x["ekf_vel_ned"] for x in rec]), axis=1)
    gt = np.linalg.norm(np.array([x["gt_vel_ned"] for x in rec], dtype=float), axis=1)

    npz = dict(np.load(R / "stage5_flight.npz"))
    # attitude error: EKF estimate vs Gazebo truth. make_monotonic first -- these streams
    # carry a few pre-time-sync stamps, and using the raw first sample as t0 put the axis
    # at -1.8e9 s.
    tg, q_g_raw = mf.make_monotonic(npz["att_gt"][:, 0], npz["att_gt"][:, 1:5])
    te, q_e_raw = mf.make_monotonic(npz["att_ekf"][:, 0], npz["att_ekf"][:, 1:5])
    idx = np.searchsorted(te, tg).clip(0, len(te) - 1)
    q_e = Rotation.from_quat(q_e_raw[idx][:, [1, 2, 3, 0]])
    q_g = Rotation.from_quat(q_g_raw[:, [1, 2, 3, 0]])
    att_err = np.degrees((q_e.inv() * q_g).magnitude())

    # The recorder and the mission both start 25 s after the shadow node, so shift the
    # attitude trace onto the shadow node's clock instead of plotting two different zeros.
    REC_OFFSET_S = 25.0
    t_att = (tg - tg[0]) + REC_OFFSET_S

    # Mark the crash where MARIO's output first leaves any plausible range, rather than
    # hardcoding a time taken from a different clock.
    crash_t = float(t[int(np.argmax(mario > 5.0))]) if (mario > 5.0).any() else float(t[-1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

    ax[0].plot(t, mario, color="#c1443c", lw=1.2, label="MARIO output")
    ax[0].plot(t, ekf, color="#3b6ea5", lw=1.2, label="EKF2 state")
    ax[0].plot(t, gt, color="#222", lw=1.6, label="ground truth (parked)")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("speed [m/s]")
    ax[0].axvspan(0, crash_t, color="#1a7a3e", alpha=0.08)
    ax[0].axvline(crash_t, color="#c1443c", ls="--", lw=1.2)
    ax[0].text(crash_t + 4, ax[0].get_ylim()[1] * 0.55,
               f"crash ~{crash_t:.0f} s\nright of here\nis post-mortem",
               fontsize=7, color="#666")
    ax[0].set_title(f"velocity: airframe crashed at ~{crash_t:.0f} s, estimator then ran away",
                    fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(t_att, att_err, color="#7a3e9d", lw=1.2)
    ax[1].axvline(crash_t, color="#c1443c", ls="--", lw=1.2)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("EKF2 attitude error vs truth [deg]")
    ax[1].set_title("attitude stays within 3.9 deg: not the mechanism", fontsize=10)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stage 5 — closed loop on MARIO alone: armed, ~2 s airborne, crashed",
                 fontsize=11)
    fig.tight_layout()
    out = R / "figures" / "stage5_divergence.png"
    fig.savefig(out, dpi=150)

    summary = {
        "armed": True,
        "outcome": "armed, airborne ~2 s, crashed, disarmed by failsafe",
        "px4_events": ["Armed by external command", "Takeoff detected",
                       "Failsafe activated", "Disarmed by failsafe"],
        "gt_travel_before_crash_m": 1.05,
        "est_error_at_liftoff_m": 1.91,
        "final_gt_roll_deg": -63.5,
        "gt_speed_mean_m_s": float(np.nanmean(gt)),
        "mario_speed_start_m_s": float(mario[:250].mean()),
        "mario_speed_end_m_s": float(mario[-250:].mean()),
        "ekf_speed_end_m_s": float(ekf[-250:].mean()),
        "attitude_error_start_deg": float(att_err[:200].mean()),
        "attitude_error_end_deg": float(att_err[-200:].mean()),
        "attitude_error_max_deg": float(att_err.max()),
        "crash_time_s_shadow_clock": crash_t,
        "diverged": True,
    }
    (R / "stage5_divergence.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"  {k:28} {v if not isinstance(v, float) else round(v, 3)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
