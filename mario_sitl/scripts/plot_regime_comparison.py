#!/usr/bin/env python
"""One model, three flight regimes: where MARIO works, degrades, and fails.

Same checkpoint throughout. The only thing that changes between panels is how fast the
airframe was moving, which is exactly the variable that decides whether velocity is present
in the IMU at all:

    drag signal = k * v,  k = 0.12 (m/s2)/(m/s) for x500
    accelerometer horizontal noise = 0.0277 m/s2
    => signal = noise at 0.23 m/s

Above that the estimate tracks; near it the estimate is noise. TDE is meaningless for a
near-stationary segment (it divides by distance travelled), so those panels report ATE and
the drift rate instead.
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
NOISE_FLOOR_M_S = 0.23


def collect(net, dev, npz: Path) -> list[dict]:
    seq = build_sequence(dict(np.load(npz)), "gt")
    pos = seq["gt_translation"].numpy()
    out = []
    for start in range(0, len(seq["acc"]) - 1030, 500):
        sl = {k: v[start:start + 1030] for k, v in seq.items()}
        rolled = rollout_trajectory(net, sl, dev)
        if rolled is None:
            continue
        gt, pred = rolled
        m = trajectory_metrics(gt, pred)
        if m is None:
            continue
        idx = np.arange(start, start + 1030)
        speed = np.linalg.norm(np.gradient(pos[idx], 0.01, axis=0), axis=1).mean()
        out.append({**m, "speed": float(speed), "start_s": start * 0.01,
                    "gt": gt, "pred": pred})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "sitl_agg" / "best.pt")
    ap.add_argument("--out", type=Path, default=R / "figures" / "regime_comparison_3d.png")
    args = ap.parse_args()

    dev = torch.device("cuda")
    net = CausalMambaDispNet().to(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=True))
    net.eval()

    segs = collect(net, dev, R / "dataset_agg" / "flight_6.npz")
    segs += collect(net, dev, R / "dataset" / "flight_6.npz")   # the gentle collection
    segs.sort(key=lambda s: s["speed"])

    def pick(lo, hi, best=True):
        band = [s for s in segs if lo <= s["speed"] < hi]
        if not band:
            return None
        return min(band, key=lambda s: s["ATE"]) if best else band[len(band) // 2]

    panels = [
        ("잘 됨 — 빠른 비행", pick(3.0, 99.0), "#1a7a3e"),
        ("적당함 — 중간 속도", pick(0.8, 2.0, best=False), "#c98a1a"),
        ("실패 — 저속·정지", pick(0.0, 0.35, best=False), "#c1443c"),
    ]
    panels = [(t, s, c) for t, s, c in panels if s is not None]

    fig = plt.figure(figsize=(4.4 * len(panels), 4.8))
    for i, (title, s, colour) in enumerate(panels):
        gt, pred = s["gt"], s["pred"]
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        ax.plot(*gt.T, color="#1a1a1a", lw=2.0, label="ground truth")
        ax.plot(*pred.T, color=colour, lw=1.8, label="MARIO (IMU only)")
        ax.scatter(*gt[0], color="#1a7a3e", s=45, depthshade=False, zorder=5)
        ax.plot(*np.stack([gt[-1], pred[-1]]).T, color="#888", ls=":", lw=1.2)

        drift = float(np.linalg.norm(pred[-1] - gt[-1]))
        tde = (f"TDE {s['TDE']:.1f} %" if s["distance"] >= 5.0
               else "TDE 무의미 (이동량 부족)")
        ax.set_title(f"{title}\n평균 속도 {s['speed']:.2f} m/s  ·  이동 {s['distance']:.1f} m\n"
                     f"ATE {s['ATE']:.2f} m  ·  {tde}\n최종 드리프트 {drift:.2f} m",
                     fontsize=9, color=colour)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle(
        "같은 모델, 같은 10.2초 구간 길이 — 속도만 다름   |   "
        f"관측 가능 한계 {NOISE_FLOOR_M_S} m/s (항력 신호 = 가속도계 잡음)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    for t, s, _ in panels:
        print(f"  {t:<22} speed {s['speed']:.2f} m/s  ATE {s['ATE']:.2f} m  "
              f"dist {s['distance']:.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
