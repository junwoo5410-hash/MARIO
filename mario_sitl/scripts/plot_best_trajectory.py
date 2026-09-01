#!/usr/bin/env python
"""3D trajectory of the best MARIO result: open loop, aggressive-trained, held-out flight.

This is MARIO at its best -- the regime where velocity is actually observable from IMU
(|accel| 4.4-4.9 m/s2, 88-97% of samples above 1 m/s2). Each panel is one independent
10.2 s rollout: displacements are chained from the true starting pose, so the drift shown
is pure dead reckoning with no absolute reference of any kind.

Note the segment length. rollout_trajectory integrates only its first window, so 10.2 s is
the horizon, not a crop for presentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Korean labels render as tofu boxes without this: matplotlib's default font has no
# Hangul coverage. NanumGothic is present on this host (fc-list confirms it).
for _f in ("NanumGothic", "NanumBarunGothic", "IPAexGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False   # keep minus signs from breaking

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from eval_openloop import build_sequence  # noqa: E402
from mario.evaluate import rollout_trajectory, trajectory_metrics  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402

R = ROOT / "mario_sitl" / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=R / "dataset_agg" / "flight_6.npz")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "sitl_agg" / "best.pt")
    ap.add_argument("--panels", type=int, default=4)
    ap.add_argument("--out", type=Path, default=R / "figures" / "best_trajectory_3d.png")
    args = ap.parse_args()

    dev = torch.device("cuda")
    net = CausalMambaDispNet().to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    net.eval()

    seq = build_sequence(dict(np.load(args.npz)), "gt")
    segs = []
    for start in range(0, len(seq["acc"]) - 1030, 1000):
        sl = {k: v[start:start + 1030] for k, v in seq.items()}
        rolled = rollout_trajectory(net, sl, dev)
        if rolled is None:
            continue
        gt, pred = rolled
        m = trajectory_metrics(gt, pred)
        if m and m["distance"] >= 5.0:
            segs.append({**m, "start_s": start * 0.01, "gt": gt, "pred": pred})
    if not segs:
        print("no segments with enough motion")
        return 1

    segs.sort(key=lambda s: s["TDE"])
    show = segs[:args.panels]
    ates = np.array([s["ATE"] for s in segs])
    tdes = np.array([s["TDE"] for s in segs])

    fig = plt.figure(figsize=(4.1 * len(show), 4.4))
    for i, s in enumerate(show):
        gt, pred = s["gt"], s["pred"]
        ax = fig.add_subplot(1, len(show), i + 1, projection="3d")
        ax.plot(*gt.T, color="#1a1a1a", lw=2.0, label="ground truth")
        ax.plot(*pred.T, color="#c1443c", lw=1.8, label="MARIO (IMU only)")
        ax.scatter(*gt[0], color="#1a7a3e", s=45, depthshade=False, label="start", zorder=5)
        ax.scatter(*gt[-1], color="#1a1a1a", s=25, depthshade=False)
        ax.scatter(*pred[-1], color="#c1443c", s=25, depthshade=False)
        # a line joining the two endpoints makes the final drift legible
        ax.plot(*np.stack([gt[-1], pred[-1]]).T, color="#888", ls=":", lw=1.2)

        ax.set_title(f"t = {s['start_s']:.0f}–{s['start_s']+10.2:.0f} s\n"
                     f"ATE {s['ATE']:.2f} m  ·  TDE {s['TDE']:.1f} %\n"
                     f"travelled {s['distance']:.1f} m", fontsize=9)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle(
        f"MARIO open-loop dead reckoning — aggressive-trained, held-out flight   "
        f"|   {len(segs)} segments: median TDE {np.median(tdes):.1f} %, "
        f"median ATE {np.median(ates):.2f} m   |   no GPS, no absolute reference",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    print(f"{len(segs)} moving segments | ATE median {np.median(ates):.3f} "
          f"best {ates.min():.3f} | TDE median {np.median(tdes):.2f} % best {tdes.min():.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
