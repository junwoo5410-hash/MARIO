#!/usr/bin/env python
"""Render a 3D GIF (or static PNG) comparing ground truth with the model rollout.

Example:
    python scripts/animate.py --split unseen --index 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mario.config import Config
from mario.data import load_eval_sequences
from mario.model import build_model
from mario.train import load_checkpoint
from mario.utils import describe_device, resolve_device, set_seed
from mario.visualize import predict_and_plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, default="configs/trial8.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None, help="override data.data_dir")
    p.add_argument("--output-dir", type=str, default=None, help="override output_dir")
    p.add_argument("--split", choices=["seen", "unseen"], default="unseen")
    p.add_argument("--index", type=int, default=1, help="1-based trajectory index within the split")
    p.add_argument("--all", action="store_true", help="render every trajectory in the split")
    p.add_argument("--frames", type=int, default=120, help="GIF frame count")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no-rotate", action="store_true", help="keep the camera fixed")
    p.add_argument("--static", action="store_true", help="write a PNG instead of a GIF")
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

    seen_seqs, unseen_seqs = load_eval_sequences(cfg.data)
    sequences = seen_seqs if args.split == "seen" else unseen_seqs
    if not sequences:
        raise SystemExit(f"No {args.split} sequences found under {cfg.data_root}")

    if args.all:
        selected = list(enumerate(sequences, start=1))
    else:
        if not 1 <= args.index <= len(sequences):
            raise SystemExit(f"--index must be within 1..{len(sequences)}")
        selected = [(args.index, sequences[args.index - 1])]

    ext = "png" if args.static else "gif"
    render_kwargs = (
        {}
        if args.static
        else {"n_frames": args.frames, "fps": args.fps, "rotate": not args.no_rotate}
    )

    for idx, (name, data) in selected:
        out_path = cfg.output_root / "figures" / f"traj_{args.split}_{idx}_{name}.{ext}"
        result = predict_and_plot(
            net,
            data,
            device,
            cfg.data,
            out_path,
            title=f"{args.split.upper()} {name}",
            animate=not args.static,
            **render_kwargs,
        )
        print(f"saved: {result}" if result else f"skipped {name}: sequence too short")


if __name__ == "__main__":
    main()
