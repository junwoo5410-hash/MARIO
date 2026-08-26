#!/usr/bin/env python
"""Stage 4b — fine-tune MARIO on SITL flights that contain the slow regime.

Stage 4 diagnosis: the network emits 0.7-1.1 m/s of phantom velocity while the airframe is
motionless, because Blackbird's minimum speed over all ten eval trajectories is 0.178 m/s
and 0.03% of samples sit below 0.2 m/s. Slow flight and hover are outside the training
distribution entirely, and our reference mission spends most of its time there.

The fix is data, not architecture: mix SITL flights (62% of samples below 0.2 m/s) into
the Blackbird set and fine-tune from the existing checkpoint at a reduced learning rate.
Blackbird is kept in the mix on purpose -- dropping it would trade the phantom-velocity
bias for a fast-flight regression.

Nothing under ``mario/`` is modified (work rule 3): this builds ``Sequence`` dicts in the
shape ``BlackbirdDispDataset`` already expects and calls ``mario.train.train`` unchanged.
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
sys.path.insert(0, str(ROOT / "mario_sitl" / "mario_ros"))

import mario_frames as mf  # noqa: E402
from eval_openloop import build_sequence  # noqa: E402
from mario.config import Config  # noqa: E402
from mario.data import load_split  # noqa: E402
from mario.dataset import BlackbirdDispDataset  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402
from mario.train import train  # noqa: E402

RESULTS = ROOT / "mario_sitl" / "results"
GROUND_CLEARANCE = 0.5  # metres above the launch surface before a sample counts as flight


def airborne_slice(seq: dict) -> dict:
    """Trim the parked head and tail of a recording.

    Ground samples are a real regime but a trivial one -- motors off, no vibration, exactly
    zero motion -- and they would dominate the loss without teaching anything about hover.
    """
    z = seq["gt_translation"][:, 2].numpy()  # NWU: up is +z
    airborne = np.where(z - z[0] > GROUND_CLEARANCE)[0]
    if len(airborne) < 2000:
        return seq
    lo, hi = int(airborne[0]), int(airborne[-1]) + 1
    return {k: v[lo:hi] for k, v in seq.items()}


def sitl_sequences(paths: list[Path], attitude: str) -> list[dict]:
    out = []
    for p in paths:
        seq = airborne_slice(build_sequence(dict(np.load(p)), attitude))
        out.append(seq)
        speed = np.linalg.norm(np.diff(seq["gt_translation"].numpy(), axis=0), axis=1) / mf.DT
        print(f"  {p.name:<16} {len(seq['acc']):>7} samples  "
              f"mean {speed.mean():.2f} m/s  under 0.2 m/s {100 * (speed < 0.2).mean():.0f}%")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=RESULTS / "dataset")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "trial8_100ep" / "best.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "sitl_finetune")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="below the 2.6e-4 used from scratch: this is a nudge, not a retrain")
    ap.add_argument("--attitude", default="gt", choices=("gt", "ekf"))
    ap.add_argument("--no-blackbird", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    flights = sorted(args.dataset.glob("flight_*.npz"))
    if not flights:
        print(f"no flights under {args.dataset}")
        return 1

    # Hold out the last flight so the test set is a whole unseen flight, not shuffled
    # windows from flights the model also trained on.
    train_paths, test_paths = flights[:-1], flights[-1:]
    print(f"SITL train ({len(train_paths)} flights):")
    train_seqs = sitl_sequences(train_paths, args.attitude)
    print(f"SITL test ({len(test_paths)} flights):")
    test_seqs = sitl_sequences(test_paths, args.attitude)

    cfg = Config()
    cfg.data.data_dir = "/src/gs25122/blackbird_data"
    if not args.no_blackbird:
        print("Blackbird sequences:")
        bb_train = [s for _, s in load_split(Path(cfg.data.data_dir), cfg.data.seen, "train",
                                             dt=cfg.data.dt, verbose=False)]
        bb_test = [s for _, s in load_split(Path(cfg.data.data_dir), cfg.data.seen, "test",
                                           dt=cfg.data.dt, verbose=False)]
        print(f"  train {len(bb_train)} sequences, test {len(bb_test)} sequences")
        train_seqs += bb_train
        test_seqs += bb_test

    train_ds = BlackbirdDispDataset(train_seqs, cfg.data.window_size,
                                    cfg.data.train_step_size, cfg.data.label_start_index,
                                    cfg.data.label_stride)
    test_ds = BlackbirdDispDataset(test_seqs, cfg.data.window_size,
                                   cfg.data.test_step_size, cfg.data.label_start_index,
                                   cfg.data.label_stride)
    print(f"windows: train {len(train_ds)}, test {len(test_ds)}")

    net = CausalMambaDispNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    print(f"loaded {args.ckpt}")

    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.output_dir = str(args.out)
    history = train(net, train_ds, test_ds, cfg, device)

    (args.out / "finetune_meta.json").write_text(json.dumps({
        "base_checkpoint": str(args.ckpt),
        "sitl_train_flights": [p.name for p in train_paths],
        "sitl_test_flights": [p.name for p in test_paths],
        "blackbird_mixed_in": not args.no_blackbird,
        "attitude_source": args.attitude,
        "epochs": args.epochs, "lr": args.lr,
        "train_windows": len(train_ds), "test_windows": len(test_ds),
        "best_test_rmse": history["best_test_rmse"],
        "elapsed_sec": history["elapsed_sec"],
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
