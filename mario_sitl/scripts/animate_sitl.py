#!/usr/bin/env python
"""Animate a SITL rollout in 3D: the drone flying its real path against MARIO's estimate.

The existing scripts/animate.py renders Blackbird eval sequences; this does the same for
recorded SITL flights, showing both the true and estimated drone position as they move so
the drift is visible as it accumulates rather than only as a final number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

for _f in ("NanumGothic", "NanumBarunGothic", "IPAexGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from eval_openloop import build_sequence  # noqa: E402
from mario.evaluate import rollout_trajectory, trajectory_metrics  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402

R = ROOT / "mario_sitl" / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=R / "dataset_yf" / "flight_5_star.npz")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "sitl_yf" / "best.pt")
    ap.add_argument("--out", type=Path, default=R / "figures" / "flight_3d.gif")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--worst", action="store_true",
                    help="pick the worst segment instead of the best")
    ap.add_argument("--min-distance", type=float, default=25.0,
                    help="only consider segments that actually travel")
    args = ap.parse_args()

    dev = torch.device("cuda")
    net = CausalMambaDispNet().to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    net.eval()

    seq = build_sequence(dict(np.load(args.npz)), "gt")
    best = None
    for st in range(0, len(seq["acc"]) - 1030, 500):
        sl = {k: v[st:st + 1030] for k, v in seq.items()}
        rolled = rollout_trajectory(net, sl, dev)
        if rolled is None:
            continue
        gt, pred = rolled
        m = trajectory_metrics(gt, pred)
        if m is None or m["distance"] < args.min_distance:
            continue
        if best is None or ((m["TDE"] > best[0]["TDE"]) if args.worst
                            else (m["TDE"] < best[0]["TDE"])):
            best = (m, gt, pred, st * 0.01)
    if best is None:
        print("no segment long enough")
        return 1
    m, gt, pred, t0 = best
    print(f"segment @ {t0:.0f}s  ATE {m['ATE']:.2f} m  TDE {m['TDE']:.2f} %  "
          f"travelled {m['distance']:.1f} m  frames {len(gt)}")

    lo = np.minimum(gt.min(0), pred.min(0)) - 1.0
    hi = np.maximum(gt.max(0), pred.max(0)) + 1.0
    span = (hi - lo).max() / 2
    mid = (hi + lo) / 2

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")

    gt_line, = ax.plot([], [], [], color="#1a1a1a", lw=2.2, label="실제 경로")
    pr_line, = ax.plot([], [], [], color="#c1443c", lw=2.0, label="MARIO 추정 (IMU만)")
    err_line, = ax.plot([], [], [], color="#999", ls=":", lw=1.4)
    gt_dot = ax.plot([], [], [], "o", color="#1a1a1a", ms=9, label="실제 위치")[0]
    pr_dot = ax.plot([], [], [], "o", color="#c1443c", ms=9, label="추정 위치")[0]
    ax.scatter(*gt[0], color="#1a7a3e", s=90, marker="*", depthshade=False, label="출발점")

    for setter, c in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(mid[c] - span, mid[c] + span)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    title = ax.set_title("")

    def update(i):
        gt_line.set_data(gt[:i+1, 0], gt[:i+1, 1]); gt_line.set_3d_properties(gt[:i+1, 2])
        pr_line.set_data(pred[:i+1, 0], pred[:i+1, 1]); pr_line.set_3d_properties(pred[:i+1, 2])
        gt_dot.set_data([gt[i, 0]], [gt[i, 1]]); gt_dot.set_3d_properties([gt[i, 2]])
        pr_dot.set_data([pred[i, 0]], [pred[i, 1]]); pr_dot.set_3d_properties([pred[i, 2]])
        seg = np.stack([gt[i], pred[i]])
        err_line.set_data(seg[:, 0], seg[:, 1]); err_line.set_3d_properties(seg[:, 2])
        d = float(np.linalg.norm(gt[i] - pred[i]))
        flown = float(np.linalg.norm(np.diff(gt[:i+1], axis=0), axis=1).sum()) if i else 0.0
        title.set_text(f"경과 {i * 0.09:.1f}s   비행거리 {flown:.1f} m   현재 오차 {d:.2f} m\n"
                       f"구간 전체:  ATE {m['ATE']:.2f} m   TDE {m['TDE']:.1f} %")
        ax.view_init(elev=24, azim=-60 + 40 * i / len(gt))   # slow orbit
        return gt_line, pr_line, err_line, gt_dot, pr_dot

    anim = FuncAnimation(fig, update, frames=len(gt), interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
