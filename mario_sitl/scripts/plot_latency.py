#!/usr/bin/env python
"""Stage 2 deliverable: end-to-end latency histogram for the MARIO shadow node."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "stage2_shadow.json"
OUT = ROOT / "results" / "figures" / "stage2_latency.png"
BUDGET_MS = 50.0  # spec Stage 2: p95 <= 50 ms


def main() -> int:
    d = json.loads(SRC.read_text())
    rec, summ = d["records"], d["summary"]
    lat = np.array([r["latency_ms"] for r in rec])
    srv = np.array([r["server_ms"] for r in rec])
    p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].hist(lat, bins=60, color="#3b6ea5", edgecolor="white", linewidth=0.4)
    ax[0].axvline(p50, color="#444", ls="--", lw=1.2, label=f"p50 {p50:.1f} ms")
    ax[0].axvline(p95, color="#c1443c", ls="--", lw=1.5, label=f"p95 {p95:.1f} ms")
    ax[0].axvline(BUDGET_MS, color="#1a7a3e", lw=1.8, label=f"budget {BUDGET_MS:.0f} ms")
    ax[0].set_xlabel("end-to-end latency [ms]\n(IMU sample arrival -> velocity in hand)")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"MARIO shadow-mode latency  (n={len(lat)}, "
                    f"{summ['achieved_rate_hz']:.1f} Hz)")
    ax[0].legend(frameon=False, fontsize=9)

    t = np.array([r["t"] for r in rec])
    ax[1].plot(t, lat, lw=0.6, color="#3b6ea5", label="end-to-end")
    ax[1].plot(t, srv, lw=0.6, color="#d19c2f", label="inference only")
    ax[1].axhline(BUDGET_MS, color="#1a7a3e", lw=1.5)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("latency [ms]")
    ax[1].set_title("latency over the run")
    ax[1].legend(frameon=False, fontsize=9)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print(f"p50 {p50:.2f} ms | p95 {p95:.2f} ms | max {lat.max():.2f} ms | "
          f"over budget {(lat > BUDGET_MS).sum()}/{len(lat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
