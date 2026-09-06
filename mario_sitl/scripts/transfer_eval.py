#!/usr/bin/env python
"""Apply a Blackbird-trained MARIO checkpoint to another dataset, unchanged.

The zero-shot baseline for the transfer study: it says how much of what the model
learned on Blackbird is about quadrotor dynamics in general and how much is about
Blackbird. Everything downstream (linear probe, partial and full fine-tuning) is
measured against the numbers this produces.

The evaluation protocol is mario.evaluate.evaluate_split verbatim -- same rollout,
same ATE = sqrt(mean(err^2)) -- so the numbers sit directly beside the Blackbird
ones in mario_sitl/results/. Nothing about the network is adapted here: no
recalibration, no bias fit, no rescaling. That is the point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config                      # noqa: E402
from mario.evaluate import evaluate_split            # noqa: E402
from mario.model import CausalMambaDispNet           # noqa: E402
from bimamba_model import BiMambaDispNet             # noqa: E402


def load_sequences(dataset: str, split: str, dt: float, verbose: bool,
                   align: str = "none", yaw_deg: float = 0.0):
    if dataset == "euroc":
        import euroc_data
        names = (euroc_data.list_sequences() if split == "all"
                 else euroc_data.read_lists()[split])
        return euroc_data.load_split(names, dt=dt, align=align, yaw_deg=yaw_deg,
                                     verbose=verbose)
    raise ValueError(f"unknown dataset {dataset!r}")


def build_net(arch: str, cfg: Config, ckpt: Path, device):
    Net = BiMambaDispNet if arch == "bi" else CausalMambaDispNet
    net = Net(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
              expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    net.eval()
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--arch", default="causal", choices=("causal", "bi"))
    ap.add_argument("--dataset", default="euroc", choices=("euroc",))
    ap.add_argument("--split", default="test", choices=("train", "val", "test", "all"),
                    help="which official split to evaluate")
    ap.add_argument("--dt", type=float, default=0.01,
                    help="resampling period; 0.01 matches the 100 Hz Blackbird "
                         "training distribution, 0.005 is EuRoC's native rate")
    ap.add_argument("--align", default="none", choices=("none", "gravity"),
                    help="rotate the body frame so hover specific force "
                         "sits where Blackbird has it (a coordinate change, not a fit)")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="extra rotation about the aligned gravity axis, which the "
                         "gravity constraint leaves free")

    ap.add_argument("--config", default=str(ROOT / "configs" / "trial8.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"checkpoint: {a.ckpt}")
    print(f"dataset:    {a.dataset} / {a.split}  dt={a.dt} ({1/a.dt:.0f} Hz)"
          + f"  align={a.align} yaw={a.yaw_deg:g}")
    seqs = load_sequences(a.dataset, a.split, a.dt, verbose=True,
                          align=a.align, yaw_deg=a.yaw_deg)

    net = build_net(a.arch, cfg, a.ckpt, device)
    print(f"arch: {a.arch}  params: {sum(p.numel() for p in net.parameters()):,}")

    title = f"ZERO-SHOT  {a.dataset}/{a.split}  dt={a.dt}  align={a.align}"
    res = evaluate_split(net, seqs, device, cfg.data, title=title)

    payload = {
        "checkpoint": str(a.ckpt), "arch": a.arch, "dataset": a.dataset,
        "split": a.split, "dt": a.dt,
        "align": a.align, "yaw_deg": a.yaw_deg, "result": res,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
