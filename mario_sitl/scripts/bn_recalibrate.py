#!/usr/bin/env python
"""Recalibrate BatchNorm running statistics on a target domain -- no gradients.

The encoders carry BatchNorm1d whose running_mean/var were fixed on Blackbird. On x500
the input distribution differs (different mass, drag, IMU noise), so those statistics are
wrong and every downstream activation is mis-centred. Recomputing them from target-domain
forward passes is the standard cheap domain-adaptation test: if the Blackbird->SITL gap is
mostly a normalisation-statistics problem, this closes much of it for free; if it is a
learned-mapping problem, this changes little. Either answer is worth 30 minutes.

Weights are untouched. Only BatchNorm buffers change.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from eval_openloop import build_sequence  # noqa: E402
from mario.config import Config  # noqa: E402
from mario.dataset import BlackbirdDispDataset, collate_fn  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "trial8_100ep" / "best.pt")
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "mario_sitl" / "results" / "dataset_yf")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "trial8_bnrecal" / "best.pt")
    ap.add_argument("--exclude", nargs="*", default=["flight_5_star.npz", "flight_8_figure8.npz"],
                    help="holdout flights -- recalibrating on them would leak the test set")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    device = torch.device("cuda")
    net = CausalMambaDispNet().to(device)
    net.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))

    flights = [f for f in sorted(a.dataset.glob("flight_*.npz")) if f.name not in a.exclude]
    print(f"recalibrating on {len(flights)} flights (excluding {a.exclude})")
    seqs = [build_sequence(dict(np.load(f)), "gt") for f in flights]
    cfg = Config()
    ds = BlackbirdDispDataset(seqs, cfg.data.window_size, cfg.data.test_step_size,
                              cfg.data.label_start_index, cfg.data.label_stride)
    print(f"  {len(ds)} windows")

    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    before = [(float(m.running_mean.mean()), float(m.running_var.mean())) for m in bns]
    print(f"  {len(bns)} BatchNorm1d layers")

    # Cumulative average (momentum=None) gives the exact mean/var over everything seen,
    # rather than an exponential window weighted towards the last batches.
    net.eval()                       # dropout off: its noise would widen the statistics
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()                    # only BN collects statistics

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))
    n = 0
    with torch.no_grad():
        for b in range(a.batches):
            sl = idx[b * a.batch_size:(b + 1) * a.batch_size]
            if len(sl) == 0:
                break
            batch = collate_fn([ds[int(i)] for i in sl])
            net(batch["acc"].to(device), batch["gyro"].to(device),
                batch["gt_rot"].to(device).Log().tensor().float())
            n += len(sl)
    print(f"  saw {n} windows over {b + 1} batches")

    after = [(float(m.running_mean.mean()), float(m.running_var.mean())) for m in bns]
    print("\n  layer   mean(before -> after)      var(before -> after)")
    for i, ((m0, v0), (m1, v1)) in enumerate(zip(before, after)):
        print(f"    {i}   {m0:+.4f} -> {m1:+.4f}      {v0:.4f} -> {v1:.4f}"
              f"   (var x{v1 / max(v0, 1e-9):.2f})")

    net.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
