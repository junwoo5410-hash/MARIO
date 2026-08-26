#!/usr/bin/env python
"""Evaluate a trained checkpoint on the SEEN and UNSEEN evaluation trajectories.

Example:
    python scripts/evaluate.py --config configs/trial8.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.config import Config
from mario.data import load_eval_sequences
from mario.evaluate import evaluate_split
from mario.model import build_model
from mario.train import load_checkpoint
from mario.utils import describe_device, resolve_device, set_seed

# Published numbers this work is compared against.
REFERENCE = {
    "velocity prediction (Optuna)": (0.523, 2.263),
    "displacement baseline": (0.429, 1.809),
    "displacement Trial 8 (30 ep)": (0.429, 1.703),
    "AirIO+Motor (bidirectional)": (0.460, 0.767),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, default="configs/trial8.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="defaults to <output_dir>/best.pt")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config if Path(args.config).expanduser().exists() else None)
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Device: {describe_device(device)}")

    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else cfg.checkpoint_path
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint} (run scripts/train.py first)")

    net = build_model(cfg.model).to(device).float()
    load_checkpoint(net, checkpoint, device)
    print(f"Loaded {checkpoint}")

    seen_seqs, unseen_seqs = load_eval_sequences(cfg.data)
    seen = evaluate_split(net, seen_seqs, device, cfg.data, title="SEEN")
    unseen = evaluate_split(net, unseen_seqs, device, cfg.data, title="UNSEEN")

    print(f"\n{'=' * 60}\n  Comparison (mean ATE [m])\n{'=' * 60}")
    print(f"{'':<32}{'SEEN':>12}{'UNSEEN':>12}")
    print("-" * 56)
    for name, (s, u) in REFERENCE.items():
        print(f"  {name:<30}{s:>12.3f}{u:>12.3f}")
    print("-" * 56)
    print(f"  {'this run':<30}{seen['mean_ate']:>12.3f}{unseen['mean_ate']:>12.3f}")
    print("=" * 60)

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    with open(cfg.results_path, "w", encoding="utf-8") as f:
        json.dump({"seen": seen, "unseen": unseen, "reference": REFERENCE}, f, indent=2)
    print(f"\nResults: {cfg.results_path}")


if __name__ == "__main__":
    main()
