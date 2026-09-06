#!/usr/bin/env python
"""Transfer ATE against the amount of EuRoC training data.

The curves cross at roughly three minutes. Below it the Blackbird weights are the only
information the network has and the linear probe wins; above it they are a liability and
training from scratch wins. Full fine-tuning is the worst option in the scarce-data
regime -- it destroys the pre-trained features before it can relearn them.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# the labels are Korean; NanumBarunGothic is what this machine has
matplotlib.rcParams["font.family"] = "NanumBarunGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "transfer" / "full_rollout_rescore.json"
OUT = ROOT / "results" / "figures" / "transfer_data_efficiency.png"

#: minutes of training data at each n (after the 20 % validation tail is removed)
MINUTES = {"euroc": {1: 1.1, 2: 2.6, 4: 6.6, 6: 9.8}}
STYLE = {"scratch": ("#c1443c", "o", "처음부터 학습"),
         "head":    ("#3b6ea5", "s", "linear probe (헤드만)"),
         "full":    ("#d19c2f", "^", "전체 파인튜닝")}



def main() -> int:
    rows = json.loads(SRC.read_text())
    by = defaultdict(list)
    zero = {}
    for r in rows:
        if r["mode"] == "zeroshot":
            zero[r["dataset"]] = r["full_ate"]
        elif r["dataset"] == "euroc" and r["select"] == "ate" and r["n_train"]:
            by[(r["dataset"], r["mode"], r["n_train"], round(r["lr"], 9))].append(r["full_ate"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ds in ("euroc",):
        # keep one lr per mode: the default the curve was run at
        lrs = {m: max({k[3] for k in by if k[0] == ds and k[1] == m}) if m == "scratch"
               else min({k[3] for k in by if k[0] == ds and k[1] == m})
               for m in STYLE}
        for mode, (c, mk, lab) in STYLE.items():
            ns = sorted(k[2] for k in by if k[0] == ds and k[1] == mode and k[3] == lrs[mode])
            if not ns:
                continue
            x = [MINUTES[ds][n] for n in ns]
            v = [np.array(by[(ds, mode, n, lrs[mode])]) for n in ns]
            m = np.array([a.mean() for a in v])
            s = np.array([a.std() for a in v])
            ax.plot(x, m, color=c, marker=mk, lw=1.6, ms=6, label=lab)
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.axhline(zero[ds], color="#555", ls="--", lw=1.3,
                   label=f"zero-shot ({zero[ds]:.1f} m)")
        ax.set_yscale("log")
        ax.set_xlabel("EuRoC 학습 데이터 [분]")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)
    ax.set_ylabel("전체 궤적 ATE [m]  (시드 4개 평균)")
    ax.set_title("사전학습을 언제 쓰고 언제 버릴 것인가\nBlackbird 학습 MARIO → EuRoC", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
