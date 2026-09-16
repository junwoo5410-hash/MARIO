#!/usr/bin/env python
"""Best-case Blackbird rollouts, one seen and one unseen trajectory.

Pure dead reckoning: the estimate is anchored to the true pose once, at the first
sample, and every position after that is the running sum of predicted body-frame
displacements. No absolute reference enters the chain again.

Trajectory choice. The lowest TDE on the seen split belongs to ``winter`` (0.573 %),
but only 48.6 % of that trajectory is evaluated -- a window needs 1000 whole samples
to start, so short sequences lose their tail, and a shorter integration flatters the
result. ``egg`` matches it on TDE (0.739 %) over four times the distance and 86 % of
the trajectory, so it is the honest picture of the same performance. ``sid`` leads
the unseen split outright at 98.5 % coverage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager                       # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402

for _f in ("NanumGothic", "NanumBarunGothic", "IPAexGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config                      # noqa: E402
from mario.data import load_split                    # noqa: E402
from mario.model import CausalMambaDispNet           # noqa: E402
from eval_full_rollout import rollout_full, metrics  # noqa: E402

GT_C, PR_C = "#2b2b2b", "#d1495b"


def roll(traj: str, seed: int, cfg, dev):
    net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                             expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(dev)
    net.load_state_dict(torch.load(ROOT / "runs" / f"nm_s{seed}" / "best.pt",
                                   map_location=dev, weights_only=True))
    _, seq = load_split("/src/gs25122/blackbird_data", [traj], "eval",
                        dt=cfg.data.dt, verbose=False)[0]
    gt, pred = rollout_full(net, seq, dev, cfg.data.window_size,
                            cfg.data.label_start_index, cfg.data.label_stride)
    full = float(np.linalg.norm(np.diff(seq["gt_translation"].numpy(), axis=0), axis=1).sum())
    return gt, pred, metrics(gt, pred), full


def draw3d(ax, gt, pred, m):
    ax.plot(*gt.T, color=GT_C, lw=1.9, label="실제 궤적 (GT)")
    ax.plot(*pred.T, color=PR_C, lw=1.9, ls="--", label="MARIO 추정")
    ax.scatter(*gt[0], color="#2a9d8f", s=55, marker="o", depthshade=False, zorder=5)
    ax.scatter(*gt[-1], color=GT_C, s=45, marker="s", depthshade=False, zorder=5)
    ax.scatter(*pred[-1], color=PR_C, s=45, marker="s", depthshade=False, zorder=5)
    ax.set_xlabel("X [m]", labelpad=-4); ax.set_ylabel("Y [m]", labelpad=-4)
    ax.set_zlabel("Z [m]", labelpad=-4)
    ax.tick_params(labelsize=7, pad=-1)
    # equal aspect so a drifting estimate cannot be hidden by axis scaling
    pts = np.vstack([gt, pred])
    c, r = pts.mean(0), (pts.max(0) - pts.min(0)).max() / 2
    for s, lo in zip("xyz", c - r):
        getattr(ax, f"set_{s}lim")(lo, lo + 2 * r)
    # the axes box is taller than the cube drawn inside it, so an "upper left" legend
    # lands next to the figure-level header; drop it into the box instead
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.03, 0.88),
              framealpha=0.92, borderpad=0.5)


def header(fig, x, tag, title, m, full):
    fig.text(x, 0.905, f"{tag}  ·  {title}", ha="center", fontsize=11.5, weight="bold")
    fig.text(x, 0.878, f"ATE {m['ATE']:.3f} m    TDE {m['TDE']:.3f} %    "
                       f"{m['distance']:.1f} m / {m['n_poses']} poses  "
                       f"(궤적의 {m['distance']/full*100:.0f} %)",
             ha="center", fontsize=10, color="#444")


def draw_err(ax, gt, pred, m, color):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    e = np.linalg.norm(pred - gt, axis=1)
    ax.plot(d, e, color=color, lw=1.6)
    ax.fill_between(d, 0, e, color=color, alpha=0.13)
    ax.axhline(m["ATE"], color=color, ls=":", lw=1.2)
    ax.text(d[-1], m["ATE"], f" ATE {m['ATE']:.3f}", va="bottom", ha="right",
            fontsize=8, color=color)
    ax.set_xlabel("이동 거리 [m]", fontsize=9)
    ax.set_ylabel("위치 오차 [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(0, d[-1]); ax.set_ylim(0, max(e.max() * 1.15, 1e-3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="egg/yawForward/maxSpeed8p0")
    ap.add_argument("--seen-seed", type=int, default=42)
    ap.add_argument("--unseen", default="sid/yawForward/maxSpeed5p0")
    ap.add_argument("--unseen-seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "mario_sitl" / "results" / "figures" / "blackbird_best.png")
    a = ap.parse_args()

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gs, ps, ms, fs = roll(a.seen, a.seen_seed, cfg, dev)
    gu, pu, mu, fu = roll(a.unseen, a.unseen_seed, cfg, dev)

    fig = plt.figure(figsize=(13.2, 9.2))
    gs_ = fig.add_gridspec(2, 2, height_ratios=[2.6, 1], hspace=0.22, wspace=0.10,
                           left=0.05, right=0.97, top=0.855, bottom=0.075)
    ax_s = fig.add_subplot(gs_[0, 0], projection="3d")
    ax_u = fig.add_subplot(gs_[0, 1], projection="3d")
    draw3d(ax_s, gs, ps, ms)
    draw3d(ax_u, gu, pu, mu)
    header(fig, 0.275, "SEEN — 학습에서 본 궤적",
           f"{a.seen.split(chr(47))[0]}  (seed {a.seen_seed})", ms, fs)
    header(fig, 0.745, "UNSEEN — 학습에 없던 궤적",
           f"{a.unseen.split(chr(47))[0]}  (seed {a.unseen_seed})", mu, fu)
    # A 3D axes reserves a wide margin inside its box for the projection; without this
    # the cubes float in whitespace and read as much smaller than the 2D panels below.
    for ax in (ax_s, ax_u):
        b = ax.get_position()
        ax.set_position([b.x0 - b.width * 0.09, b.y0 - b.height * 0.10,
                         b.width * 1.18, b.height * 1.20])
    draw_err(fig.add_subplot(gs_[1, 0]), gs, ps, ms, "#2b6cb0")
    draw_err(fig.add_subplot(gs_[1, 1]), gu, pu, mu, "#d1495b")

    fig.suptitle("MARIO (NM, causal, 76,582 params) — Blackbird 최고 성능 롤아웃\n"
                 "GT 첫 위치에 한 번만 앵커, 이후 예측 변위만 누적 (순수 dead reckoning)",
                 fontsize=13, y=0.982)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"seen   {a.seen.split(chr(47))[0]:8} ATE {ms['ATE']:.3f}  TDE {ms['TDE']:.3f} %  "
          f"{ms['distance']:.1f}/{fs:.1f} m")
    print(f"unseen {a.unseen.split(chr(47))[0]:8} ATE {mu['ATE']:.3f}  TDE {mu['TDE']:.3f} %  "
          f"{mu['distance']:.1f}/{fu:.1f} m")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
