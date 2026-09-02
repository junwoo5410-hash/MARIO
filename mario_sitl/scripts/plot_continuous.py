#!/usr/bin/env python
"""Plot the continuous-movement run: does drift stay bounded when the airframe keeps moving?

Stage 5/6 asked the vehicle to hold position, which is the one thing an IMU-only estimator
cannot do, and every run diverged at 0.17-1.36 m/s. This run never stops, and GPS is cut in
flight rather than absent from the start.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

for _f in ("NanumGothic", "NanumBarunGothic", "IPAexGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False

R = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=R / "continuous_run.json")
    ap.add_argument("--out", type=Path, default=R / "figures" / "continuous_run.png")
    a = ap.parse_args()

    d = json.loads(a.run.read_text())
    s, ev = d["summary"], d["events"]
    rows = [x for x in d["samples"]
            if not (np.isnan(np.asarray(x["gt"], dtype=float)).any()
                    or np.isnan(np.asarray(x["est"], dtype=float)).any())]
    t = np.array([x["t"] for x in rows]); t -= t[0]
    gt = np.array([x["gt"] for x in rows], dtype=float)
    est = np.array([x["est"] for x in rows], dtype=float)
    sp = np.array([x["setpoint"] for x in rows], dtype=float)
    ph = [x["phase"] for x in rows]

    gt_d = gt - np.array(s["gt_origin"], dtype=float)
    est_d = est - np.array(s["ekf_origin"], dtype=float)
    sp_d = sp - np.array(s["ekf_origin"], dtype=float)
    err = np.linalg.norm(est_d - gt_d, axis=1)

    cut_i = next((i for i, p in enumerate(ph) if p == "lap_mario"), len(ph) - 1)
    t_cut = t[cut_i]

    fig = plt.figure(figsize=(16.5, 5.0))
    pre, post = slice(0, cut_i), slice(cut_i, None)

    # 3D view. NED z is down-positive, so negate it to plot altitude upward.
    ax3 = fig.add_subplot(1, 4, 1, projection="3d")
    ax3.plot(sp_d[:, 0], sp_d[:, 1], -sp_d[:, 2], color="#1a7a3e", ls="--", lw=1.0,
             label="명령 궤적(셋포인트)")
    ax3.plot(gt_d[pre, 0], gt_d[pre, 1], -gt_d[pre, 2], color="#999", lw=2.0,
             label="실제 위치 — GPS 사용 구간")
    ax3.plot(gt_d[post, 0], gt_d[post, 1], -gt_d[post, 2], color="#1a1a1a", lw=2.0,
             label="실제 위치 — GPS 차단 후")
    ax3.plot(est_d[post, 0], est_d[post, 1], -est_d[post, 2], color="#c1443c", lw=1.5,
             label="EKF2 추정 (MARIO 속도 적분)")
    ax3.scatter(gt_d[cut_i, 0], gt_d[cut_i, 1], -gt_d[cut_i, 2], color="#c98a1a", s=80,
                marker="X", depthshade=False, zorder=6, label="GPS 차단")
    ax3.set_xlabel("N [m]", fontsize=8); ax3.set_ylabel("E [m]", fontsize=8)
    ax3.set_zlabel("고도 [m]", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.set_title("3D 궤적", fontsize=10)
    ax3.legend(fontsize=7, frameon=False, loc="upper left")

    ax = [fig.add_subplot(1, 4, i) for i in (2, 3, 4)]

    # horizontal track
    ax[0].plot(sp_d[:, 0], sp_d[:, 1], color="#1a7a3e", ls="--", lw=1.0, label="명령 궤적(셋포인트)")
    ax[0].plot(gt_d[pre, 0], gt_d[pre, 1], color="#999", lw=2.0, label="실제 위치 — GPS 사용 구간")
    ax[0].plot(gt_d[post, 0], gt_d[post, 1], color="#1a1a1a", lw=2.0, label="실제 위치 — GPS 차단 후")
    ax[0].plot(est_d[post, 0], est_d[post, 1], color="#c1443c", lw=1.5, label="EKF2 추정 (MARIO 속도 적분)")
    ax[0].scatter(gt_d[cut_i, 0], gt_d[cut_i, 1], color="#c98a1a", s=70, zorder=5,
                  marker="X", label="GPS 차단")
    ax[0].set_xlabel("North [m]"); ax[0].set_ylabel("East [m]")
    ax[0].set_aspect("equal", "datalim")
    ax[0].set_title("수평 궤적 (위에서)", fontsize=10)
    ax[0].legend(fontsize=7.5, frameon=False)

    # error over time
    ax[1].plot(t, err, color="#c1443c", lw=1.4)
    ax[1].axvline(t_cut, color="#c98a1a", ls="--", lw=1.4)
    ax[1].text(t_cut + 1, err.max() * 0.15, "GPS 차단", fontsize=8, color="#c98a1a")
    ax[1].set_xlabel("시간 [s]"); ax[1].set_ylabel("추정 오차 |est − gt| [m]")
    ax[1].set_title(f"추정 오차 — 차단 후 증가율 {s['drift_rate_m_per_s']:.3f} m/s", fontsize=10)

    # comparison against the station-keeping runs
    prev = [0.17, 0.33, 0.55, 0.80, 0.80, 0.92, 1.09, 1.13, 1.23, 1.36]
    ax[2].boxplot([prev], positions=[0], widths=0.5)
    ax[2].scatter([1], [s["drift_rate_m_per_s"]], color="#1a7a3e", s=110, zorder=5)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["제자리 유지 미션\n(Stage 6, n=10)", "연속 이동\n(이번 런)"],
                          fontsize=8.5)
    ax[2].set_ylabel("추정 오차 증가율 [m/s]")
    ax[2].set_yscale("log")
    ax[2].set_title("발산률 비교 (로그 축)", fontsize=10)
    for a_ in ax[1:]:
        a_.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"연속 이동 폐루프 — GPS 차단 후 {s['mario_only_seconds']:.0f}초, "
        f"{s['distance_flown_m']:.0f} m 비행   |   최종 추정 오차 {s['est_error_final_m']:.1f} m "
        f"= 이동거리의 {s['est_error_pct_of_distance']:.1f} %", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=155)
    print(f"wrote {a.out}")
    print(f"차단 후 {s['mario_only_seconds']:.1f} s, {s['distance_flown_m']:.0f} m")
    print(f"최종 오차 {s['est_error_final_m']:.2f} m ({s['est_error_pct_of_distance']:.2f} %)")
    print(f"증가율 {s['drift_rate_m_per_s']:.4f} m/s  vs Stage 6 중앙값 {np.median(prev):.2f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
