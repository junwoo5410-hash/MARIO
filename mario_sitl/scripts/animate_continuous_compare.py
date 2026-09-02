#!/usr/bin/env python
"""Side-by-side 3D animation of two continuous-movement closed-loop runs.

Same circle, same speed, same GPS-cut procedure -- only the MARIO checkpoint differs, so
everything visible between the two panels is attributable to the network. Both panels are
drawn on one shared cube so the size difference between the two flown paths is real and
not an artifact of per-panel autoscaling.

The GPS phase is drawn in grey and the MARIO-only phase in black; the estimate appears
only after the cut, because before it EKF2 is on GPS and the estimate is not being tested.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

for _f in ("NanumGothic", "NanumBarunGothic", "IPAexGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False

R = Path(__file__).resolve().parents[1] / "results"


def load(p: Path):
    """Return NWU-style arrays (N, E, up) so altitude plots upward."""
    d = json.loads(p.read_text())
    s = d["summary"]
    rows = [x for x in d["samples"]
            if not (np.isnan(np.asarray(x["gt"], float)).any()
                    or np.isnan(np.asarray(x["est"], float)).any())]
    t = np.array([x["t"] for x in rows]); t -= t[0]
    gt = np.array([x["gt"] for x in rows], float) - np.array(s["gt_origin"], float)
    est = np.array([x["est"] for x in rows], float) - np.array(s["ekf_origin"], float)
    sp = np.array([x["setpoint"] for x in rows], float) - np.array(s["ekf_origin"], float)
    ph = [x["phase"] for x in rows]
    cut = next((i for i, p_ in enumerate(ph) if p_ == "lap_mario"), len(ph) - 1)
    for a in (gt, est, sp):
        a[:, 2] *= -1.0                      # NED down -> altitude up
    return dict(s=s, t=t, gt=gt, est=est, sp=sp, cut=cut)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=R / "continuous_run.json")
    ap.add_argument("--b", type=Path, default=R / "continuous_run_yf.json")
    ap.add_argument("--label-a", default="Blackbird 학습 (trial8)")
    ap.add_argument("--label-b", default="SITL yawForward 학습 (sitl_yf)")
    ap.add_argument("--out", type=Path, default=R / "figures" / "continuous_compare_3d.gif")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--tail", type=float, default=12.0,
                    help="seconds of trail kept behind the drone; 0 keeps everything")
    a = ap.parse_args()

    D = [load(a.a), load(a.b)]
    labs = [a.label_a, a.label_b]

    # Shared limits across both panels so the size difference between the flown paths is
    # real. Horizontal axes share one span (a circle must look circular); altitude gets
    # its own, because the flight spans ~6 m vertically against ~45 m horizontally and a
    # cube would flatten both trajectories into an unreadable pancake. The z axis is
    # compressed via box_aspect instead, which keeps the distortion visible as such.
    pts = np.concatenate([np.concatenate([d["gt"], d["est"]]) for d in D])
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) / 2
    mid = (hi + lo) / 2
    zlo, zhi = lo[2], hi[2]

    fig = plt.figure(figsize=(12.4, 5.8))
    fig.patch.set_facecolor("white")
    axes, art = [], []
    for k in range(2):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        d = D[k]
        ax.plot(d["sp"][:, 0], d["sp"][:, 1], d["sp"][:, 2],
                color="#1a7a3e", ls="--", lw=1.0, label="명령 궤적 (반경 8 m)")
        h = dict(
            gps=ax.plot([], [], [], color="#bbb", lw=1.8, label="실제 — GPS 구간")[0],
            gt=ax.plot([], [], [], color="#1a1a1a", lw=2.0, label="실제 — GPS 차단 후")[0],
            est=ax.plot([], [], [], color="#c1443c", lw=1.8, label="EKF2 추정 (MARIO)")[0],
            link=ax.plot([], [], [], color="#888", ls=":", lw=1.2)[0],
            gtd=ax.plot([], [], [], "o", color="#1a1a1a", ms=8)[0],
            estd=ax.plot([], [], [], "o", color="#c1443c", ms=8)[0],
        )
        ax.scatter(*d["gt"][d["cut"]], color="#c98a1a", s=90, marker="X",
                   depthshade=False, label="GPS 차단 지점")
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(zlo, zhi)
        ax.set_box_aspect((1.0, 1.0, 0.42))
        ax.set_xlabel("North [m]", fontsize=8); ax.set_ylabel("East [m]", fontsize=8)
        ax.set_zlabel("고도 [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        # lower left: the upper left is where the running metrics in the title land
        ax.legend(loc="lower left", fontsize=7.2, frameon=False,
                  bbox_to_anchor=(0.0, 0.02))
        axes.append(ax); art.append(h)

    titles = [ax.set_title("", fontsize=9.5, y=1.0) for ax in axes]
    sup = fig.suptitle("", fontsize=12)

    n = max(len(d["t"]) - d["cut"] for d in D)
    step = max(1, n // a.frames)

    def update(f):
        j = f * step
        for k, (d, h) in enumerate(zip(D, art)):
            i = min(d["cut"] + j, len(d["t"]) - 1)
            tail = 0
            if a.tail > 0:
                tail = int(np.searchsorted(d["t"], d["t"][i] - a.tail))
                tail = max(tail, d["cut"])
            h["gps"].set_data(d["gt"][:d["cut"], 0], d["gt"][:d["cut"], 1])
            h["gps"].set_3d_properties(d["gt"][:d["cut"], 2])
            for key, src in (("gt", d["gt"]), ("est", d["est"])):
                h[key].set_data(src[tail:i + 1, 0], src[tail:i + 1, 1])
                h[key].set_3d_properties(src[tail:i + 1, 2])
            seg = np.stack([d["gt"][i], d["est"][i]])
            h["link"].set_data(seg[:, 0], seg[:, 1]); h["link"].set_3d_properties(seg[:, 2])
            h["gtd"].set_data([d["gt"][i, 0]], [d["gt"][i, 1]])
            h["gtd"].set_3d_properties([d["gt"][i, 2]])
            h["estd"].set_data([d["est"][i, 0]], [d["est"][i, 1]])
            h["estd"].set_3d_properties([d["est"][i, 2]])

            err = float(np.linalg.norm(d["gt"][i] - d["est"][i]))
            flown = float(np.linalg.norm(np.diff(d["gt"][d["cut"]:i + 1], axis=0),
                                         axis=1).sum()) if i > d["cut"] else 0.0
            spd = flown / max(d["t"][i] - d["t"][d["cut"]], 1e-6)
            titles[k].set_text(f"{labs[k]}\n현재 오차 {err:5.2f} m   "
                               f"이동 {flown:5.1f} m   평균 속도 {spd:4.2f} m/s")
            axes[k].view_init(elev=26, azim=-62 + 46 * f / max(a.frames - 1, 1))
        el = D[0]["t"][min(D[0]["cut"] + j, len(D[0]["t"]) - 1)] - D[0]["t"][D[0]["cut"]]
        sup.set_text(f"GPS 차단 후 {el:4.1f} s   —   명령 속도 3.5 m/s, 반경 8 m 원")
        return []

    frames = min(a.frames, (n + step - 1) // step)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / a.fps, blit=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.02)
    anim.save(a.out, writer=PillowWriter(fps=a.fps))
    print(f"wrote {a.out}  ({a.out.stat().st_size / 1e6:.1f} MB, {frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
