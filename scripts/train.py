#!/usr/bin/env python
"""Train the Causal Mamba displacement network.

Example:
    python scripts/train.py --config configs/trial8.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.config import Config
from mario.data import load_training_sequences
from mario.dataset import build_datasets
from mario.model import build_model
from mario.train import train
from mario.utils import count_parameters, describe_device, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, default="configs/trial8.yaml", help="path to a YAML config")
    p.add_argument("--data-dir", type=str, default=None, help="override data.data_dir")
    p.add_argument("--output-dir", type=str, default=None, help="override output_dir")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    p.add_argument("--device", type=str, default=None, help="cuda, cpu or auto")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config if Path(args.config).expanduser().exists() else None)

    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Device: {describe_device(device)}")

    train_seqs, test_seqs = load_training_sequences(cfg.data)
    if not train_seqs:
        raise SystemExit(f"No training sequences found under {cfg.data_root}")

    print("Building windows and displacement labels...")
    train_ds, test_ds = build_datasets(train_seqs, test_seqs, cfg.data)
    print(f"train windows: {len(train_ds)} | test windows: {len(test_ds)}")

    net = build_model(cfg.model).to(device).float()
    print(f"Parameters: {count_parameters(net):,}")

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.output_root / "config.yaml")

    summary = train(net, train_ds, test_ds, cfg, device)

    with open(cfg.output_root / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"History: {cfg.output_root / 'training_history.json'}")


if __name__ == "__main__":
    main()
