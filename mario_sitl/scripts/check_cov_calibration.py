#!/usr/bin/env python
"""Is the covariance head calibrated? EKF2 consumes it as measurement noise, so it matters.

The head is trained with a Gaussian NLL on a detached residual at weight 1e-4. If it is
calibrated, E[residual^2 / cov] == 1 and the two are correlated across samples. If it is
not, EKF2 is weighting MARIO's velocity by a number that means nothing.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from eval_openloop import build_sequence  # noqa: E402
from mario.config import Config  # noqa: E402
from mario.data import load_split  # noqa: E402
from mario.dataset import BlackbirdDispDataset  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402


def stats(name, res2, cov):
    """res2 and cov are (N, 3) per-axis squared residual and predicted variance."""
    z = res2 / np.maximum(cov, 1e-12)
    print(f"\n--- {name} ---")
    print(f"  samples {len(res2)}")
    print(f"  RMS residual   {np.sqrt(res2.mean()):.5f} m per 0.09 s step")
    print(f"  mean sqrt(cov) {np.sqrt(cov.mean()):.5f} m")
    # if the head has collapsed to a constant this spread is ~0 and every downstream
    # correlation is undefined, which is exactly what a nan corr indicates
    print(f"  cov spread     min {cov.min():.3e}  max {cov.max():.3e}  "
          f"std/mean {cov.std() / max(cov.mean(), 1e-12):.3e}")
    print(f"  E[res^2/cov]   {z.mean():8.3f}   (1.0 = calibrated)")
    print(f"  median         {np.median(z):8.3f}")
    # does a larger predicted variance actually mean a larger error?
    for ax in range(3):
        c = np.corrcoef(cov[:, ax], res2[:, ax])[0, 1]
        sc = np.corrcoef(np.argsort(np.argsort(cov[:, ax])),
                         np.argsort(np.argsort(res2[:, ax])))[0, 1]
        print(f"  axis {ax}: corr(cov, res^2) = {c:+.3f}   spearman = {sc:+.3f}")
    # decile test: sort by predicted variance, does actual error rise with it?
    order = np.argsort(cov.mean(1))
    dec = np.array_split(order, 10)
    pred = [np.sqrt(cov[d].mean()) for d in dec]
    act = [np.sqrt(res2[d].mean()) for d in dec]
    print("  decile  예측 sigma / 실제 RMS:")
    for i, (p, a) in enumerate(zip(pred, act)):
        print(f"    {i}  {p:.5f}  {a:.5f}   ratio {a / max(p, 1e-9):6.2f}")


def collect(net, ds, device, limit=4000):
    res2, cov = [], []
    idx = np.linspace(0, len(ds) - 1, min(limit, len(ds))).astype(int)
    with torch.no_grad():
        for i in idx:
            b = ds[int(i)]
            t = {k: b[k].unsqueeze(0).to(device) for k in ("acc", "gyro", "motor")}
            # same rotation handling as mario.train._forward_batch
            rot = b["gt_rot"].unsqueeze(0).to(device).Log().tensor().float()
            d, c = net(t["acc"], t["gyro"], rot, t["motor"])
            gt = b["gt_disp"].unsqueeze(0).to(device)
            n = min(d.shape[1], gt.shape[1])
            res2.append(((d[:, :n] - gt[:, :n]) ** 2).squeeze(0).cpu().numpy())
            cov.append(c[:, :n].squeeze(0).cpu().numpy())
    return np.concatenate(res2), np.concatenate(cov)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "trial8_100ep" / "best.pt")
    ap.add_argument("--sitl", type=Path,
                    default=ROOT / "mario_sitl" / "results" / "dataset_yf")
    ap.add_argument("--skip-sitl", action="store_true",
                    help="Blackbird only -- the SITL pass is 4x the samples and this "
                         "comparison is about the Blackbird-domain calibration")
    a = ap.parse_args()

    device = torch.device("cuda")
    net = CausalMambaDispNet().to(device)
    net.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))
    net.eval()
    print(f"checkpoint {a.ckpt}")

    cfg = Config()
    cfg.data.data_dir = "/src/gs25122/blackbird_data"
    bb = [s for _, s in load_split(Path(cfg.data.data_dir), cfg.data.seen, "test",
                                  dt=cfg.data.dt, verbose=False)]
    ds = BlackbirdDispDataset(bb, cfg.data.window_size, cfg.data.test_step_size,
                              cfg.data.label_start_index, cfg.data.label_stride)
    stats("Blackbird test (자기 도메인)", *collect(net, ds, device))

    flights = [] if a.skip_sitl else sorted(a.sitl.glob("flight_*.npz"))[:2]
    if flights:
        seqs = [build_sequence(dict(np.load(f)), "gt") for f in flights]
        ds2 = BlackbirdDispDataset(seqs, cfg.data.window_size, cfg.data.test_step_size,
                                   cfg.data.label_start_index, cfg.data.label_stride)
        stats(f"SITL ({len(flights)} flights, 도메인 밖)", *collect(net, ds2, device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
