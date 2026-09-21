#!/usr/bin/env python
"""Pick the yaw of the UZH-FPV body alignment, on training sequences only.

Gravity alignment fixes two of the three body-frame degrees of freedom; rotation about
the gravity axis is left free, and on EuRoC it was worth 15.19 m -> 2.61 m of zero-shot
ATE. The same sweep is needed here. Scoring happens on the TRAIN split: the test
sequences must not be touched by a choice that every later number depends on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config                       # noqa: E402
from mario.model import CausalMambaDispNet            # noqa: E402
from eval_full_rollout import rollout_full, metrics   # noqa: E402
import uzhfpv_data                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "nm_s42" / "best.pt")
    ap.add_argument("--step", type=float, default=30.0)
    ap.add_argument("--yaws", type=float, nargs="+", default=None,
                    help="explicit yaw values, for refining around a coarse-sweep minimum")
    ap.add_argument("--align", default="gravity", choices=("yaw", "gravity"),
                    help="which rotation family to sweep")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "mario_sitl" / "results" / "uzhfpv" / "yaw_sweep.json")
    a = ap.parse_args()

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                             expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
    net.eval()

    names = uzhfpv_data.read_lists()["train"]
    rows = []
    print(f"{'yaw':>6}{'full ATE':>11}{'full TDE':>11}   (train split, {len(names)} seqs)")
    for yaw in (np.array(a.yaws) if a.yaws else np.arange(0.0, 360.0, a.step)):
        pairs = uzhfpv_data.load_split(names, dt=0.01, align=a.align,
                                       yaw_deg=float(yaw), verbose=False)
        per = {}
        for name, seq in pairs:
            out = rollout_full(net, seq, dev, cfg.data.window_size,
                               cfg.data.label_start_index, cfg.data.label_stride)
            if out is not None:
                per[name] = metrics(*out)
        ate = float(np.mean([m["ATE"] for m in per.values()]))
        tde = float(np.mean([m["TDE"] for m in per.values()]))
        rows.append({"yaw_deg": float(yaw), "ate": ate, "tde": tde,
                     "per_sequence": {k: v["ATE"] for k, v in per.items()}})
        print(f"{yaw:>6.0f}{ate:>11.3f}{tde:>11.2f}")

    # the "none" reference, so the value of aligning at all is on the record
    pairs = uzhfpv_data.load_split(names, dt=0.01, align="none", verbose=False)
    per = {}
    for name, seq in pairs:
        out = rollout_full(net, seq, dev, cfg.data.window_size,
                           cfg.data.label_start_index, cfg.data.label_stride)
        if out is not None:
            per[name] = metrics(*out)
    unaligned = float(np.mean([m["ATE"] for m in per.values()]))

    best = min(rows, key=lambda r: r["ate"])
    payload = {"checkpoint": str(a.ckpt), "split": "train", "align": a.align,
               "sequences": names,
               "unaligned_ate": unaligned, "best": best, "sweep": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"\nno alignment      {unaligned:8.3f}")
    print(f"{a.align}, yaw {best['yaw_deg']:>3.0f}  {best['ate']:8.3f}  <- pick")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
