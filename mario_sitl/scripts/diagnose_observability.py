#!/usr/bin/env python
"""Is velocity observable in this data, and does the model recover it?

Two measurements that between them explain every result in Stage 4 and 5.

1. OBSERVABILITY (data only, no model). IMU reveals velocity through exactly two
   channels: transient dynamics, which can be integrated over short spans, and
   steady-state drag, a horizontal specific force proportional to speed. A vehicle
   cruising at constant velocity with weak drag emits almost nothing about its speed.
   Measured: Blackbird flies at |accel| 4.6-6.8 m/s2 with 98-100% of moving samples above
   1 m/s2 and a drag signal of 1.07-1.86 m/s2. The first SITL collection, which walked the
   position setpoint at a commanded speed, managed 0.42-0.67 m/s2, 13-26%, and 0.13-0.18.
   Both channels an order of magnitude down.

2. DIRECTION (model). Cosine similarity between predicted and true body-frame
   displacement, with the magnitude ratio alongside it. This separates "right direction,
   wrong scale" -- which a per-airframe scalar could fix -- from "wrong direction", which
   nothing downstream can fix. Blackbird in-domain scores +0.828; the SITL-trained model on
   its own held-out flight scored -0.002, which is what sent the investigation back to the
   data rather than to the training recipe.

Usage:
    diagnose_observability.py --data 'results/dataset_agg/*.npz' --ckpt <path>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from eval_openloop import build_sequence  # noqa: E402
from mario.data import load_blackbird  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402

GRAVITY = 9.80665
MOVING_M_S = 0.5


def load_any(path: str) -> dict:
    if path.endswith(".npz"):
        return build_sequence(dict(np.load(path)), "gt")
    return load_blackbird(os.path.dirname(path))


def observability(seq: dict) -> dict | None:
    acc = seq["acc"].numpy()
    R = seq["gt_orientation"].matrix().numpy()
    pos = seq["gt_translation"].numpy()

    aw = np.einsum("tij,tj->ti", R, acc)
    aw[:, 2] -= GRAVITY                       # world acceleration, gravity removed
    v = np.gradient(pos, 0.01, axis=0)
    sp = np.linalg.norm(v, axis=1)
    m = sp > MOVING_M_S
    if m.sum() < 300:
        return None
    a = np.linalg.norm(aw[m], axis=1)
    return {
        "speed_mean": float(sp[m].mean()),
        "accel_mean": float(a.mean()),
        "accel_over_1_pct": float(100 * (a > 1).mean()),
        "drag_signal": float(np.linalg.norm(acc[m][:, :2], axis=1).mean()),
        "moving_samples": int(m.sum()),
    }


def direction(seq: dict, net, device) -> dict | None:
    gt_pos = seq["gt_translation"].numpy()
    gt_rot = seq["gt_orientation"]
    P, T = [], []
    for start in range(0, len(seq["acc"]) - 1030, 1000):
        sl = {k: v[start:start + 1030] for k, v in seq.items()}
        with torch.no_grad():
            d, _ = net(sl["acc"].unsqueeze(0).to(device), sl["gyro"].unsqueeze(0).to(device),
                       gt_rot[start:start + 1030].unsqueeze(0).to(device).Log().tensor().float())
        P.append(d[0].cpu().numpy()[:112])
        idx = 14 + 9 * np.arange(112)
        tw = gt_pos[start + idx + 9] - gt_pos[start + idx]
        T.append(np.einsum("tji,tj->ti", gt_rot[start + idx].matrix().numpy(), tw))
    if not P:
        return None
    P, T = np.concatenate(P), np.concatenate(T)
    m = np.linalg.norm(T, axis=1) / 0.09 > MOVING_M_S
    if m.sum() < 50:
        return None
    P, T = P[m], T[m]
    cos = (P * T).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-9)
    return {
        "cosine": float(cos.mean()),
        "magnitude_ratio": float(np.median(np.linalg.norm(P, axis=1)
                                           / (np.linalg.norm(T, axis=1) + 1e-9))),
        "corr_x_lateral": float(np.corrcoef(P[:, 0], T[:, 0])[0, 1]),
        "corr_y_fwdback": float(np.corrcoef(P[:, 1], T[:, 1])[0, 1]),
        "corr_z_updown": float(np.corrcoef(P[:, 2], T[:, 2])[0, 1]),
        "samples": int(m.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/dataset_agg/*.npz",
                    help="glob of .npz recordings or Blackbird imu_data.csv paths")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="omit to run the data-only observability half")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.data))[:args.limit]
    if not paths:
        print(f"no files matched {args.data}")
        return 1

    net = None
    if args.ckpt:
        device = torch.device("cuda")
        net = CausalMambaDispNet().to(device)
        net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
        net.eval()

    print(f"{'file':<22}{'speed':>7}{'|accel|':>9}{'>1 m/s2':>9}{'drag':>8}", end="")
    print(f"{'cos':>8}{'|p|/|t|':>9}{'x/y/z corr':>20}" if net else "")
    print("-" * (55 + (37 if net else 0)))

    rows = {}
    for p in paths:
        seq = load_any(p)
        o = observability(seq)
        if o is None:
            continue
        name = os.path.basename(p if p.endswith(".npz") else os.path.dirname(p))[:21]
        line = (f"{name:<22}{o['speed_mean']:>7.2f}{o['accel_mean']:>9.2f}"
                f"{o['accel_over_1_pct']:>8.0f}%{o['drag_signal']:>8.3f}")
        row = {"observability": o}
        if net:
            d = direction(seq, net, device)
            if d:
                line += (f"{d['cosine']:>+8.3f}{d['magnitude_ratio']:>9.2f}"
                         f"{d['corr_x_lateral']:>+7.2f}{d['corr_y_fwdback']:>+7.2f}"
                         f"{d['corr_z_updown']:>+6.2f}")
                row["direction"] = d
        print(line)
        rows[name] = row

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
