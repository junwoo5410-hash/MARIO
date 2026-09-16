#!/usr/bin/env python
"""Trajectory grid for the current model: Blackbird seen on top, unseen below.

Each panel is one full-trajectory rollout produced by eval_full_rollout.rollout_full --
the window steps by 999 so consecutive windows' labels meet and the chain spans the
sequence, rather than mario.evaluate.rollout_trajectory's first window alone. The
estimate is anchored once at the true starting pose and every later position is the
previous one plus a predicted body-frame displacement, so what the panels show is pure
dead reckoning with no absolute reference.

Blackbird has five trajectories per split and the grid holds four, so the shortest
sequence in each split is dropped (star, 16.0 s; sphinx, 19.9 s). Pass --all for the
2x5 grid that keeps them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

for _f in ("NanumGothic", "NanumBarunGothic", "IPAexGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config, DEFAULT_SEEN, DEFAULT_UNSEEN  # noqa: E402
from mario.data import load_split                              # noqa: E402
from mario.model import CausalMambaDispNet                     # noqa: E402
from eval_full_rollout import rollout_full, metrics            # noqa: E402

#: dropped for the 2x4 layout -- the shortest sequence in each split
DROP = {"star", "sphinx"}

GT_STYLE = dict(color="0.45", lw=1.8, ls="--", label="GT")
PR_STYLE = dict(color="#0b6fa4", lw=1.8, label="MARIO")


def best_azim(gt: np.ndarray) -> float:
    """Look along the trajectory's narrow horizontal axis, so its shape reads widest.

    A fixed azimuth flattens whichever trajectory happens to be edge-on to it -- oval and
    halfMoon collapse to near-lines at azim=-58. The first principal component of the
    ground-truth ground track gives the direction of greatest spread; viewing 90 degrees
    off it puts that spread across the panel.
    """
    xy = gt[:, :2] - gt[:, :2].mean(0)
    if not np.isfinite(xy).all() or np.allclose(xy, 0):
        return -58.0
    v = np.linalg.svd(xy, full_matrices=False)[2][0]
    return float(np.degrees(np.arctan2(v[1], v[0])) - 90.0)


def panel(ax, gt: np.ndarray, pred: np.ndarray, name: str, m: dict,
          text: bool = True) -> None:
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], **GT_STYLE)
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], **PR_STYLE)
    ax.scatter(*gt[0], color="k", s=22, zorder=5)          # shared anchor

    both = np.concatenate([gt, pred])
    lo, hi = both.min(0), both.max(0)
    pad = 0.06 * np.maximum(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # Limits follow each axis's own range and the box is stretched to match, so a metre is
    # the same length on all three axes -- equal scaling without the empty volume a cubic
    # box leaves above and below a mostly-horizontal flight.
    # zoom fills the subplot the axes was given: without titles and tick labels there is
    # nothing else competing for that space, so the curves can be drawn larger.
    ax.set_box_aspect(tuple(span / span.max()), zoom=1.18 if not text else 1.0)

    if text:
        ax.set_title(f"{name}\nATE {m['ATE']:.2f} m  ·  TDE {m['TDE']:.2f} %",
                     fontsize=9, pad=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        if not text:
            axis.set_ticklabels([])
    ax.tick_params(labelsize=6, pad=-2, length=0 if not text else 3.5)
    ax.view_init(elev=24, azim=best_azim(gt))
    ax.grid(alpha=0.25)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "nm_s42" / "best.pt")
    ap.add_argument("--data-root", default="/src/gs25122/blackbird_data")
    ap.add_argument("--all", action="store_true", help="2x5 grid, keep every trajectory")
    ap.add_argument("--no-text", action="store_true",
                    help="curves only: no titles, tick labels, row labels or legend")
    ap.add_argument("--pick", nargs="+", metavar="NAME",
                    help="plot only these trajectories, in one row, in this order")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "mario_sitl" / "results" / "figures" / "trajectory_grid.png")
    a = ap.parse_args()

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                             expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))

    if a.pick:
        wanted = {n.lower(): i for i, n in enumerate(a.pick)}
        rows = (("", [t for t in list(DEFAULT_SEEN) + list(DEFAULT_UNSEEN)
                      if t.split("/")[0].lower() in wanted]),)
        rows[0][1].sort(key=lambda t: wanted[t.split("/")[0].lower()])
        missing = set(wanted) - {t.split("/")[0].lower() for t in rows[0][1]}
        if missing:
            print(f"unknown trajectory: {', '.join(sorted(missing))}")
            return 1
        ncol, nrow = len(rows[0][1]), 1
    else:
        rows = (("seen — 학습 궤적", DEFAULT_SEEN), ("unseen — 미관측 궤적", DEFAULT_UNSEEN))
        ncol, nrow = (5 if a.all else 4), 2
    fig = plt.figure(figsize=(3.5 * ncol,
                              (3.3 if a.no_text else 4.6) * nrow))
    summary = {}

    for r, (label, trajs) in enumerate(rows):
        kept, ates, tdes = [], [], []
        for t in trajs:
            name, seq = load_split(a.data_root, [t], "eval", dt=cfg.data.dt,
                                   verbose=False)[0]
            if not (a.all or a.pick) and name in DROP:
                continue
            out = rollout_full(net, seq, dev, cfg.data.window_size,
                               cfg.data.label_start_index, cfg.data.label_stride)
            if out is None:
                print(f"[skip] {name}: rollout produced no chain")
                continue
            gt, pred = out
            m = metrics(gt, pred)
            kept.append((name, gt, pred, m))
            ates.append(m["ATE"]); tdes.append(m["TDE"])

        for c, (name, gt, pred, m) in enumerate(kept):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            panel(ax, gt, pred, name, m, text=not a.no_text)
            if c == 0 and not a.no_text and label:
                ax.text2D(-0.15, 0.5, label, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=11, fontweight="bold")
        summary[label.split(" ")[0] or "picked"] = {"mean_ate": float(np.mean(ates)),
                                        "mean_tde": float(np.mean(tdes)),
                                        "trajectories": [k[0] for k in kept]}
        print(f"{label or 'picked'}: mean ATE {np.mean(ates):.3f} m, "
              f"mean TDE {np.mean(tdes):.2f} %")

    if a.no_text:
        fig.tight_layout(rect=(0.0, 0.0, 1, 1), pad=0.4)
        fig.subplots_adjust(hspace=0.06, wspace=0.06)
    else:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.008), fontsize=10.5)
        fig.suptitle(f"MARIO 전체 궤적 dead reckoning — {a.ckpt.parent.name} "
                     f"({sum(p.numel() for p in net.parameters()):,} params)",
                     fontsize=13, y=0.985)
        fig.tight_layout(rect=(0.015, 0.035, 1, 0.95))
    # 3D axes reserve little room for their tick labels, so tight_layout leaves the
    # lower row's titles sitting on the upper row's z-axis numbers.
    if not a.no_text:
        fig.subplots_adjust(hspace=0.28)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
