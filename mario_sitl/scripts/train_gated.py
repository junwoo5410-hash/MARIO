#!/usr/bin/env python
"""Train the control (CausalMambaDispNet) and the gated variant under identical conditions.

Everything except the model class comes from train_blackbird_v2.py -- the same window
cache, the same seen/unseen lists, the same checkpoint selection, the same evaluate()
protocol -- so the only thing that differs between the two arms is the fusion layer.
No extra hyper-parameter search is run for the gated arm (work rule 3 of the brief).

Usage: train_gated.py --arch gated --seed 42 --out runs/gated/gated_s42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config                       # noqa: E402
from mario.model import CausalMambaDispNet            # noqa: E402
from mario.model_gated import GatedMambaDispNet       # noqa: E402
from mario.train import train                         # noqa: E402
import train_blackbird_v2 as bb                       # noqa: E402

ARCHS = {"control": CausalMambaDispNet, "gated": GatedMambaDispNet}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=tuple(ARCHS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="default: config (100)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "trial8.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    cfg.data.data_dir = str(Path(cfg.data.data_dir).expanduser())
    if a.epochs:
        cfg.train.epochs = a.epochs
    cfg.output_dir = str(a.out)
    cfg.seed = a.seed
    bb.set_seed(cfg.seed)
    device = torch.device("cuda")
    root = Path(cfg.data.data_dir)

    # identical data path to train_blackbird_v2: same loader, same cache keys
    tag = "nomotor"
    tr_key = f"{tag}|{','.join(sorted(t.split('/')[0] for t in cfg.data.seen))}"
    train_ds = bb.build(bb.load_seqs(root, "train", list(cfg.data.seen)),
                        f"train_{tag}", cfg, cfg.data.train_step_size, tr_key)
    test_ds = bb.build(bb.load_seqs(root, "test", list(cfg.data.seen)),
                       f"test_{tag}", cfg, cfg.data.test_step_size, tag)

    net = ARCHS[a.arch](d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                        expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"arch: {a.arch}  seed: {a.seed}  params: {n_params:,}")

    t0 = time.time()
    hist = train(net, train_ds, test_ds, cfg, device)
    net.load_state_dict(torch.load(Path(cfg.output_dir) / "best.pt",
                                   map_location=device, weights_only=True))
    net.eval()
    res = bb.evaluate(net, cfg, device)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=2))
    (a.out / "run_meta.json").write_text(json.dumps({
        "arch": a.arch, "n_params": n_params, "seed": cfg.seed,
        "epochs": cfg.train.epochs, "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "huber_delta": cfg.train.huber_delta, "loss_weight": cfg.train.loss_weight,
        "uncertainty_weight": cfg.train.uncertainty_weight,
        "best_test_rmse": hist["best_test_rmse"],
        "elapsed_sec": time.time() - t0,
    }, indent=2))
    print(f"{a.arch} s{a.seed}: seen ATE {res['seen']['mean_ate']:.4f}  "
          f"unseen ATE {res['unseen']['mean_ate']:.4f}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
