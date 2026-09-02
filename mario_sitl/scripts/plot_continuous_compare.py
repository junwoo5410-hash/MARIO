#!/usr/bin/env python
"""Overlay two continuous-movement runs that differ only in the MARIO checkpoint.

Everything else in the flight is identical -- same circle, same speed, same GPS-cut
procedure -- so any difference in the drift curve is attributable to the network.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

for _f in ("NanumGothic", "NanumBarunGothic", "IPAexGothic"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False

R = Path(__file__).resolve().parents[1] / "results"


def load(p: Path):
    d = json.loads(p.read_text())
    s = d["summary"]
    rows = [x for x in d["samples"]
            if not (np.isnan(np.asarray(x["gt"], dtype=float)).any()
                    or np.isnan(np.asarray(x["est"], dtype=float)).any())]
    t = np.array([x["t"] for x in rows]); t -= t[0]
    gt = np.array([x["gt"] for x in rows], dtype=float) - np.array(s["gt_origin"], float)
    est = np.array([x["est"] for x in rows], dtype=float) - np.array(s["ekf_origin"], float)
    sp = np.array([x["setpoint"] for x in rows], dtype=float) - np.array(s["ekf_origin"], float)
    ph = [x["phase"] for x in rows]
    cut = next((i for i, p in enumerate(ph) if p == "lap_mario"), len(ph) - 1)
    err = np.linalg.norm(est - gt, axis=1)
    dev = np.linalg.norm(gt - sp, axis=1)
    return dict(s=s, t=t, gt=gt, est=est, sp=sp, cut=cut, err=err, dev=dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=R / "continuous_run.json")
    ap.add_argument("--b", type=Path, default=R / "continuous_run_yf.json")
    ap.add_argument("--label-a", default="Blackbird 학습 (trial8)")
    ap.add_argument("--label-b", default="SITL yawForward 학습 (sitl_yf)")
    ap.add_argument("--out", type=Path, default=R / "figures" / "continuous_compare.png")
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    CA, CB = "#8a8a8a", "#c1443c"
    # per-panel estimate colour: panel A cannot reuse the grey it draws GPS track in
    EST = {0: "#2b5fa8", 1: "#c1443c"}

    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.3))

    # 1-2: horizontal track, one per run, ground truth vs estimate after the cut
    for k, (D, lab, col) in enumerate(((A, a.label_a, CA), (B, a.label_b, CB))):
        post = slice(D["cut"], None)
        ax[k].plot(D["sp"][:, 0], D["sp"][:, 1], color="#1a7a3e", ls="--", lw=1.0,
                   label="명령 궤적")
        ax[k].plot(D["gt"][:D["cut"], 0], D["gt"][:D["cut"], 1], color="#bbb", lw=1.8,
                   label="실제 — GPS 구간")
        ax[k].plot(D["gt"][post, 0], D["gt"][post, 1], color="#1a1a1a", lw=1.8,
                   label="실제 — GPS 차단 후")
        ax[k].plot(D["est"][post, 0], D["est"][post, 1], color=EST[k], lw=1.6,
                   label="EKF2 추정 (MARIO)")
        ax[k].scatter(D["gt"][D["cut"], 0], D["gt"][D["cut"], 1], color="#c98a1a",
                      marker="X", s=70, zorder=5, label="GPS 차단")
        ax[k].set_aspect("equal"); ax[k].set_xlim(-30, 30); ax[k].set_ylim(-16, 34)
        ax[k].set_xlabel("North [m]"); ax[k].set_ylabel("East [m]")
        ax[k].set_title(f"{lab}\n오차 {D['s']['est_error_final_m']:.1f} m "
                        f"= 거리의 {D['s']['est_error_pct_of_distance']:.2f} %", fontsize=9.5)
        ax[k].legend(fontsize=6.8, frameon=False)

    # 3: drift curves on a common clock starting at the GPS cut
    for D, lab, col in ((A, a.label_a, CA), (B, a.label_b, CB)):
        post = slice(D["cut"], None)
        tt = D["t"][post] - D["t"][D["cut"]]
        ax[2].plot(tt, D["err"][post], color=col, lw=1.7,
                   label=f"{lab}  ({D['s']['drift_rate_m_per_s']:.4f} m/s)")
    ax[2].set_xlabel("GPS 차단 후 경과 [s]"); ax[2].set_ylabel("추정 오차 |est - gt| [m]")
    ax[2].set_title("드리프트 성장", fontsize=10)
    ax[2].legend(fontsize=7.5, frameon=False)

    # 4: the three headline numbers side by side
    keys = [("est_error_pct_of_distance", "거리 대비\n최종 오차 [%]"),
            ("drift_rate_m_per_s", "드리프트율\n[m/s]"),
            ("vehicle_deviation_max_m", "기체 이탈\n최대 [m]")]
    x = np.arange(len(keys)); w = 0.36
    va = [A["s"][k] for k, _ in keys]; vb = [B["s"][k] for k, _ in keys]
    ax[3].bar(x - w / 2, va, w, color=CA, label=a.label_a)
    ax[3].bar(x + w / 2, vb, w, color=CB, label=a.label_b)
    for xi, (u, v) in enumerate(zip(va, vb)):
        ax[3].text(xi - w / 2, u, f"{u:.3g}", ha="center", va="bottom", fontsize=7.5)
        ax[3].text(xi + w / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7.5)
    ax[3].set_xticks(x); ax[3].set_xticklabels([n for _, n in keys], fontsize=8.5)
    ax[3].set_yscale("log")
    # the log formatter renders exponents with U+2212, which NanumGothic lacks
    ax[3].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax[3].yaxis.set_minor_formatter(mticker.NullFormatter())
    ax[3].set_title("핵심 지표 (로그 축)", fontsize=10)
    ax[3].legend(fontsize=7.5, frameon=False)

    for a_ in ax[2:]:
        a_.spines[["top", "right"]].set_visible(False)

    fig.suptitle("연속 이동 폐루프 — 체크포인트만 바꾼 동일 비행", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=155)
    print(f"wrote {a.out}")

    hdr = f"{'지표':<26}{a.label_a:>26}{a.label_b:>30}"
    print(hdr); print("-" * len(hdr))
    for k in ("mario_only_seconds", "distance_flown_m", "est_error_final_m",
              "est_error_max_m", "est_error_pct_of_distance",
              "vehicle_deviation_max_m", "drift_rate_m_per_s"):
        print(f"{k:<26}{A['s'][k]:>26.4f}{B['s'][k]:>30.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
