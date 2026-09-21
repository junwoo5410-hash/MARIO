#!/usr/bin/env python
"""Stationary baseline and zero-shot diagnostics for UZH-FPV.

Two questions the yaw sweep raised:

  * what does "predict nothing" score on these sequences? Nothing above that number is
    evidence of anything, and on EuRoC no arm ever cleared it;
  * is the unaligned zero-shot (15.39 m, better than any aligned yaw) actually tracking
    the motion, or is it just outputting near-zero displacement and inheriting the
    baseline? The magnitude ratio |pred| / |true| separates those.
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
from stationary_baseline import ZeroDisp              # noqa: E402
import uzhfpv_data                                    # noqa: E402

RESULTS = ROOT / "mario_sitl" / "results" / "uzhfpv"


def score(pairs, net, dev, cfg) -> dict:
    per = {}
    for name, seq in pairs:
        out = rollout_full(net, seq, dev, cfg.data.window_size,
                           cfg.data.label_start_index, cfg.data.label_stride)
        if out is not None:
            per[name] = metrics(*out)
    return {"per_trajectory": per,
            "mean_ate": float(np.mean([m["ATE"] for m in per.values()])),
            "mean_tde": float(np.mean([m["TDE"] for m in per.values()]))}


def magnitude_ratio(net, pairs, dev, cfg) -> float:
    """Median |predicted displacement| / |true displacement| over moving windows."""
    d = cfg.data
    ratios = []
    for _, seq in pairs:
        acc, gyro = seq["acc"], seq["gyro"]
        rot = seq["gt_orientation"]
        pos = seq["gt_translation"].numpy()
        for start in range(0, len(acc) - d.window_size - 30, d.window_size):
            sl = slice(start, start + d.window_size)
            with torch.no_grad():
                pred, _ = net(acc[sl].unsqueeze(0).to(dev), gyro[sl].unsqueeze(0).to(dev),
                              rot[sl].unsqueeze(0).to(dev).Log().tensor().float())
            p = pred[0].cpu().numpy()[:112]
            idx = d.label_start_index + d.label_stride * np.arange(112)
            tw = pos[start + idx + d.label_stride] - pos[start + idx]
            t = np.einsum("tji,tj->ti", rot[start + idx].matrix().numpy(), tw)
            m = np.linalg.norm(t, axis=1) / (d.label_stride * d.dt) > 0.5
            if m.sum():
                ratios.append(np.linalg.norm(p[m], axis=1) / (np.linalg.norm(t[m], axis=1) + 1e-9))
    return float(np.median(np.concatenate(ratios))) if ratios else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "nm_s42" / "best.pt")
    ap.add_argument("--yaw", type=float, default=210.0, help="best yaw from the sweep")
    a = ap.parse_args()

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                             expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
    net.eval()
    zero = ZeroDisp(net).to(dev).eval()

    lists = uzhfpv_data.read_lists()
    out = {}
    for split in ("train", "test"):
        variants = {"none": dict(align="none", yaw_deg=0.0),
                    f"gravity_yaw{a.yaw:g}": dict(align="gravity", yaw_deg=a.yaw)}
        for label, kw in variants.items():
            pairs = uzhfpv_data.load_split(lists[split], dt=0.01, verbose=False, **kw)
            base = score(pairs, zero, dev, cfg)
            shot = score(pairs, net, dev, cfg)
            ratio = magnitude_ratio(net, pairs, dev, cfg)
            out[f"{split}/{label}"] = {
                "stationary_ate": base["mean_ate"], "stationary_tde": base["mean_tde"],
                "zeroshot_ate": shot["mean_ate"], "zeroshot_tde": shot["mean_tde"],
                "zeroshot_magnitude_ratio": ratio,
                "per_trajectory_stationary": {k: v["ATE"] for k, v in base["per_trajectory"].items()},
                "per_trajectory_zeroshot": {k: v["ATE"] for k, v in shot["per_trajectory"].items()},
            }
            print(f"{split:<6} {label:<18} stationary {base['mean_ate']:7.2f} m | "
                  f"zero-shot {shot['mean_ate']:7.2f} m | |pred|/|true| {ratio:.2f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
