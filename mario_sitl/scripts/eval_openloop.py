#!/usr/bin/env python
"""Stage 4 — open-loop MARIO accuracy on a recorded SITL flight.

Takes the raw ``.npz`` from ``mario_ros/record_flight_node.py``, puts it on the training
grid, maps it into MARIO's frames, and runs the unmodified
``mario.evaluate.rollout_trajectory`` so the number is comparable with the Blackbird ATE.

Two attitude variants are scored:
  gt   -- Gazebo ground-truth attitude, matching how the network was trained (함정 C)
  ekf  -- the EKF2 estimate, which is all a closed-loop run can supply

The gap between them is the price of that mismatch, measured rather than assumed.

Caveat from the spec: SITL IMU carries no propeller vibration, so it is cleaner than a
real airframe. These numbers do not transfer to hardware unchanged.

Second caveat, found while running this stage: ``rollout_trajectory`` integrates only the
FIRST window of whatever sequence it is handed, so a 150 s flight yields 10.2 s of
trajectory. That function extends ``positions`` by ``t_next = t_start + 9`` and only when
``t_start`` is already a key, but ``window_size % label_stride == 1000 % 9 == 1``, so
window k's label indices sit at ``(5 + k) mod 9`` and never land on window k-1's keys.
Measured directly: 14994 input samples -> 113 poses.

That is existing MARIO behaviour and ``mario/`` is off limits (work rule 3), so this
script slices the flight into consecutive 10 s segments and calls the unmodified function
once per segment. It also means the Blackbird ATE figures are per-10 s-segment numbers,
not whole-trajectory ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pypose as pp  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "mario_ros"))

import mario_frames as mf  # noqa: E402
from mario.evaluate import rollout_trajectory, trajectory_metrics  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402

RESULTS = ROOT / "mario_sitl" / "results"


def build_sequence(npz: dict, attitude: str) -> dict:
    """Raw PX4 topics -> the dict rollout_trajectory expects, on the 100 Hz training grid."""
    imu = npz["imu"]
    att = npz["att_gt" if attitude == "gt" else "att_ekf"]
    pos = npz["pos_gt"]

    imu_t, gyro_frd, acc_frd = mf.make_monotonic(imu[:, 0], imu[:, 1:4], imu[:, 4:7])
    att_t, quat = mf.make_monotonic(att[:, 0], att[:, 1:5])
    pos_t, pos_ned = mf.make_monotonic(pos[:, 0], pos[:, 1:4])

    t0 = max(imu_t[0], att_t[0], pos_t[0])
    t1 = min(imu_t[-1], att_t[-1], pos_t[-1])
    grid = np.arange(t0, t1 - mf.DT, mf.DT)

    gyro, acc, R_ned_frd = mf.resample(imu_t, gyro_frd, acc_frd, att_t, quat, grid)

    from scipy.interpolate import interp1d
    pos_g = interp1d(pos_t, pos_ned, axis=0)(np.clip(grid, pos_t[0], pos_t[-1]))

    # PX4 -> MARIO: NED->NWU world, FRD->B_M body (Stage 3 verified).
    acc_m = np.einsum("ij,tj->ti", mf.C_B, acc)
    gyro_m = np.einsum("ij,tj->ti", mf.C_B, gyro)
    R_m = np.einsum("ij,tjk,kl->til", mf.C_W, R_ned_frd, mf.C_B.T)
    pos_nwu = np.einsum("ij,tj->ti", mf.C_W, pos_g)

    from scipy.spatial.transform import Rotation
    quat_xyzw = Rotation.from_matrix(R_m).as_quat()


    return {
        "acc": torch.tensor(acc_m, dtype=torch.float32),
        "gyro": torch.tensor(gyro_m, dtype=torch.float32),
        "gt_orientation": pp.SO3(torch.tensor(quat_xyzw, dtype=torch.float32)),
        "gt_translation": torch.tensor(pos_nwu, dtype=torch.float32),
    }


def plot(results: dict, out: Path) -> None:
    fig = plt.figure(figsize=(11, 4.5))
    for i, (name, r) in enumerate(results.items()):
        # Show the segment that travelled furthest: the most demanding one.
        seg = max(r["per_segment"], key=lambda g: g["distance"])
        gt, pred = np.array(seg["gt_xyz"]), np.array(seg["pred_xyz"])
        ax = fig.add_subplot(1, len(results), i + 1, projection="3d")
        ax.plot(*gt.T, color="#333", lw=1.4, label="ground truth")
        ax.plot(*pred.T, color="#c1443c", lw=1.4, label=f"MARIO ({name} attitude)")
        ax.scatter(*gt[0], color="#1a7a3e", s=30, label="start")
        ax.set_title(f"{name} attitude — longest segment @ {seg['start_s']:.0f}s\n"
                     f"ATE {seg['ATE']:.3f} m | travelled {seg['distance']:.1f} m\n"
                     f"all moving segments: mean ATE {r['ATE']:.3f} m", fontsize=9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=RESULTS / "stage4_flight.npz")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "trial8_100ep" / "best.pt")
    ap.add_argument("--out", type=Path, default=RESULTS / "stage4_openloop.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = CausalMambaDispNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()

    npz = dict(np.load(args.npz))
    print(f"{args.npz.name}: " + ", ".join(f"{k}={v.shape[0]}" for k, v in npz.items()))

    # One rollout call per segment. 1030 samples: > 1000 so the window exists, and > 1022
    # so every one of the 112 steps still satisfies the function's t_next < seq_len guard.
    SEG, STEP, MOVING_M = 1030, 1000, 1.0

    results = {}
    for attitude in ("gt", "ekf"):
        seq = build_sequence(npz, attitude)
        n = len(seq["acc"])
        segments = []
        for start in range(0, n - SEG, STEP):
            sl = {k: v[start:start + SEG] for k, v in seq.items()}
            rolled = rollout_trajectory(net, sl, device)
            if rolled is None:
                continue
            gt_xyz, pred_xyz = rolled
            m = trajectory_metrics(gt_xyz, pred_xyz)
            if m is None:
                continue
            segments.append({**m, "start_s": start * mf.DT,
                             "gt_xyz": gt_xyz.tolist(), "pred_xyz": pred_xyz.tolist()})
        if not segments:
            print(f"{attitude}: no usable segments")
            continue

        moving = [g for g in segments if g["distance"] >= MOVING_M]
        static = [g for g in segments if g["distance"] < MOVING_M]
        agg = {
            "segments": len(segments),
            "moving_segments": len(moving),
            "static_segments": len(static),
            "ATE": float(np.mean([g["ATE"] for g in moving])) if moving else float("nan"),
            "ATE_max": float(np.max([g["ATE"] for g in moving])) if moving else float("nan"),
            "TDE": float(np.mean([g["TDE"] for g in moving])) if moving else float("nan"),
            # Segments with the airframe essentially parked: pure drift, TDE undefined.
            "static_drift_ATE": float(np.mean([g["ATE"] for g in static])) if static else 0.0,
            "duration_s": n * mf.DT,
        }
        results[attitude] = {**agg, "per_segment": segments}
        print(f"{attitude:>4} attitude: ATE {agg['ATE']:.3f} m (max {agg['ATE_max']:.3f}) | "
              f"TDE {agg['TDE']:.2f} % | {len(moving)} moving / {len(static)} static segments "
              f"| static drift {agg['static_drift_ATE']:.3f} m")

    if not results:
        print("no usable windows -- record a longer flight")
        return 1

    plot(results, RESULTS / "figures" / "stage4_openloop.png")

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "per_segment"}
               for k, v in results.items()}
    if "gt" in summary and "ekf" in summary:
        summary["attitude_penalty_m"] = summary["ekf"]["ATE"] - summary["gt"]["ATE"]
    detail = {k: [{kk: vv for kk, vv in g.items() if kk not in ("gt_xyz", "pred_xyz")}
                   for g in v["per_segment"]] for k, v in results.items()}
    args.out.write_text(json.dumps({"summary": summary, "per_segment": detail}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
