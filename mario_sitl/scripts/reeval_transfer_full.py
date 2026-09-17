#!/usr/bin/env python
"""Re-score every transfer run over the whole target trajectory.

The runs were scored with mario.evaluate, whose window chaining drops everything after
the first window (see eval_full_rollout.py). On Blackbird that hid 30-50 % of each
flight; on EuRoC, where sequences run 100-180 s, it left about 10 s -- under a tenth.
The arm-to-arm comparison was still fair, since every arm was scored the same way, but
the absolute numbers understated drift and the ranking had never been checked at the
horizon that matters. This rescores each saved checkpoint with the continuous rollout.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config                       # noqa: E402
from mario.model import CausalMambaDispNet            # noqa: E402
from eval_full_rollout import rollout_full, metrics   # noqa: E402
import euroc_data                                     # noqa: E402


def load_test(dataset, align, yaw):
    return euroc_data.load_split(euroc_data.read_lists()["test"], dt=0.01,
                                 align=align, yaw_deg=yaw, verbose=False)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "transfer")
    ap.add_argument("--zeroshot", type=Path, nargs="+",
                    default=[ROOT / "runs" / "m_s42" / "best.pt"],
                    help="un-tuned source checkpoints scored as the zero-shot reference")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "mario_sitl" / "results" / "transfer" / "full_rollout_rescore.json")
    args = ap.parse_args()

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}
    rows = []

    runs = sorted(args.runs_dir.glob("*/results.json"))
    for i, p in enumerate(runs, 1):
        r = json.loads(p.read_text())
        ck = p.parent / "best.pt"
        if not ck.exists():
            continue
        key = (r["dataset"], r["align"], r["yaw_deg"])
        if key not in cache:
            cache[key] = load_test(*key)
        net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                                 expand=cfg.model.expand,
                                 num_layers=cfg.model.num_layers).to(dev)
        net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        per = {}
        for name, seq in cache[key]:
            out = rollout_full(net, seq, dev, cfg.data.window_size,
                               cfg.data.label_start_index, cfg.data.label_stride)
            if out is not None:
                per[name] = metrics(*out)
        rows.append({
            "run": p.parent.name, "dataset": r["dataset"], "mode": r["mode"],
            "select": r.get("select", "rmse"), "lr": r["lr"], "seed": r["seed"],
            "n_train": r.get("n_train_seqs"),
            "windowed_ate": r["result"]["mean_ate"],
            "full_ate": float(np.mean([m["ATE"] for m in per.values()])),
            "full_tde": float(np.mean([m["TDE"] for m in per.values()])),
        })
        print(f"[{i:>2}/{len(runs)}] {p.parent.name:<28} "
              f"windowed {rows[-1]['windowed_ate']:7.3f} -> full {rows[-1]['full_ate']:8.3f}")

    # zero-shot reference, same rollout
    ds, yaw = "euroc", 185.0
    for ck in args.zeroshot:
        net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                                 expand=cfg.model.expand,
                                 num_layers=cfg.model.num_layers).to(dev)
        net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        per = {}
        for name, seq in load_test(ds, "gravity", yaw):
            out = rollout_full(net, seq, dev, cfg.data.window_size,
                               cfg.data.label_start_index, cfg.data.label_stride)
            if out is not None:
                per[name] = metrics(*out)
        rows.append({"run": f"{ds}_zeroshot_{ck.parent.name}", "dataset": ds, "mode": "zeroshot",
                     "select": "-", "lr": 0.0, "seed": ck.parent.name, "n_train": 0,
                     "init": str(ck), "windowed_ate": None,
                     "full_ate": float(np.mean([m["ATE"] for m in per.values()])),
                     "full_tde": float(np.mean([m["TDE"] for m in per.values()]))})
        print(f"zero-shot {ck.parent.name:<22} full {rows[-1]['full_ate']:8.3f}")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["select"], r["mode"], round(r["lr"], 9), r["n_train"])].append(r)
    order = {"zeroshot": 0, "scratch": 1, "head": 2, "trunk": 3, "full": 4}
    for ds in sorted({k[0] for k in by}):
        print(f"\n=== {ds}: 전체 궤적 기준 ===")
        print(f"{'select':<8}{'mode':<10}{'lr':>10}{'n_tr':>5}{'seeds':>7}"
              f"{'full ATE':>11}{'+-':>8}{'full TDE':>11}{'windowed':>11}")
        ks = sorted((k for k in by if k[0] == ds),
                    key=lambda k: (k[1], -k[3], order.get(k[2], 9), k[4] or 0))
        for k in ks:
            g = by[k]
            a = np.array([x["full_ate"] for x in g])
            t = np.array([x["full_tde"] for x in g])
            w = [x["windowed_ate"] for x in g if x["windowed_ate"] is not None]
            print(f"{k[1]:<8}{k[2]:<10}{k[3]:>10.3g}{k[4] or '-':>5}{len(g):>7}"
                  f"{a.mean():>11.3f}{a.std():>8.3f}{t.mean():>11.2f}"
                  f"{(np.mean(w) if w else float('nan')):>11.3f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
