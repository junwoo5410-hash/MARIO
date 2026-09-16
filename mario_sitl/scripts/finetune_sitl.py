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
import hashlib
import json
import re
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
        # The flight never really left the ground. Returning it untrimmed (the old
        # behaviour) fed a whole recording of stationary samples into training, which
        # teaches exactly the "always answer near zero" bias this project spent three
        # fine-tuning runs removing. One collected flight failed to arm and slipped in
        # this way.
        return None
    lo, hi = int(airborne[0]), int(airborne[-1]) + 1
    return {k: v[lo:hi] for k, v in seq.items()}


def sitl_sequences(paths: list[Path], attitude: str) -> list[dict]:
    out = []
    for p in paths:
        seq = airborne_slice(build_sequence(dict(np.load(p)), attitude))
        if seq is None:
            print(f"  {p.name:<22} SKIPPED — never got airborne")
            continue
        out.append(seq)
        speed = np.linalg.norm(np.diff(seq["gt_translation"].numpy(), axis=0), axis=1) / mf.DT
        print(f"  {p.name:<22} {len(seq['acc']):>7} samples  "
              f"mean {speed.mean():.2f} m/s  under 0.2 m/s {100 * (speed < 0.2).mean():.0f}%")
    return out


def balance_by_speed(ds, bins: int = 8, seed: int = 0):
    """Subsample windows so the speed histogram is flat.

    The collected flights are 57% below 0.2 m/s, and training straight on that taught the
    network to answer "near zero": the parked phantom velocity fell from 1.06 to 0.21 m/s
    while the cruise underestimate grew from 40% to 51%. One bias traded for another,
    exactly as the sampling dictated. Flattening the histogram asks the network to be
    right across the range instead of right about the most common case.
    """
    import torch.utils.data as Data

    speed = np.array([float(w["gt_disp"].norm(dim=-1).mean()) / 0.09 for w in ds.windows])

    # Equal-WIDTH bins over the speed range. Quantile edges would be equal-count by
    # construction, so equalising them is a no-op -- the first attempt at this "balanced"
    # 38357 windows down to 38352 and changed nothing.
    edges = np.linspace(speed.min(), speed.max() + 1e-9, bins + 1)
    idx_by_bin = [np.where((speed >= edges[i]) & (speed < edges[i + 1]))[0]
                  for i in range(bins)]
    idx_by_bin = [b for b in idx_by_bin if len(b)]

    # Level to the median bin: subsample the crowded low-speed bins, oversample the rare
    # fast ones. Levelling to the minimum instead would throw away most of an already
    # small dataset.
    target = int(np.median([len(b) for b in idx_by_bin]))
    rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.choice(b, target, replace=len(b) < target)
                           for b in idx_by_bin])
    rng.shuffle(keep)
    hist = [len(b) for b in idx_by_bin]
    print(f"  speed-balanced: {len(ds)} -> {len(keep)} windows, "
          f"{len(idx_by_bin)} equal-width bins levelled to {target}")
    print(f"    original per-bin counts: {hist}")
    print(f"    bin edges [m/s]: {[round(float(e), 2) for e in edges]}")
    return Data.Subset(ds, keep.tolist())


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
    ap.add_argument("--no-sitl", action="store_true",
                    help="train on Blackbird only -- the reverse transfer direction, "
                         "starting from a SITL-tuned checkpoint")
    ap.add_argument("--holdout", type=int, default=2,
                    help="whole flights held out for test, spread across the collection")
    ap.add_argument("--balance-speed", action="store_true",
                    help="flatten the training speed histogram (see balance_by_speed)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch/numpy; the loader shuffle was unseeded, so runs were "
                         "not reproducible and a single run could not be told from noise")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda")
    # Numeric sort: names like flight_10_zigzag sort before flight_2_figure8 lexically,
    # which would silently pick a different holdout than intended.
    def _idx(f: Path) -> int:
        m = re.search(r"flight_(\d+)", f.name)
        return int(m.group(1)) if m else 0

    if args.no_blackbird and args.no_sitl:
        print("--no-blackbird and --no-sitl leave no training data")
        return 1

    flights = [] if args.no_sitl else sorted(args.dataset.glob("flight_*.npz"), key=_idx)
    if not flights and not args.no_sitl:
        print(f"no flights under {args.dataset}")
        return 1

    # Hold out whole flights, not shuffled windows: adjacent windows share 997 of their
    # 1000 samples, so a shuffled split would put near-duplicates on both sides. Spread the
    # holdout across the list so it spans several trajectory shapes rather than one.
    if flights:
        k = max(1, min(args.holdout, len(flights) - 1))
        step = max(1, len(flights) // (k + 1))
        held = {flights[min(step * (i + 1), len(flights) - 1)] for i in range(k)}
        test_paths = [f for f in flights if f in held]
        train_paths = [f for f in flights if f not in held]
        print(f"SITL train ({len(train_paths)} flights):")
        train_seqs = sitl_sequences(train_paths, args.attitude)
        print(f"SITL test ({len(test_paths)} flights):")
        test_seqs = sitl_sequences(test_paths, args.attitude)
    else:
        test_paths, train_paths = [], []
        train_seqs, test_seqs = [], []
        print("SITL data excluded (--no-sitl)")

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

    # Window construction runs 111 pypose ops per window (70k windows -> 7.8M), which took
    # 40 minutes of CPU on this dataset. Cache it: the inputs are fully determined by the
    # flight list, the attitude source and the window parameters.
    def build_or_load(seqs, paths, step, tag):
        key = hashlib.md5(
            f"{[p.name for p in paths]}|{args.attitude}|{cfg.data.window_size}|{step}|"
            f"{cfg.data.label_start_index}|{cfg.data.label_stride}|"
            f"bb={not args.no_blackbird}|sitl={not args.no_sitl}".encode()).hexdigest()[:12]
        cache = args.dataset / f".windows_{tag}_{key}.pt"
        if cache.exists():
            ds = BlackbirdDispDataset([], cfg.data.window_size, step,
                                      cfg.data.label_start_index, cfg.data.label_stride)
            ds.windows = torch.load(cache, weights_only=False)
            print(f"  {tag}: {len(ds)} windows from cache")
            return ds
        ds = BlackbirdDispDataset(seqs, cfg.data.window_size, step,
                                  cfg.data.label_start_index, cfg.data.label_stride)
        torch.save(ds.windows, cache)
        return ds

    train_ds = build_or_load(train_seqs, train_paths, cfg.data.train_step_size, "train")
    test_ds = build_or_load(test_seqs, test_paths, cfg.data.test_step_size, "test")
    print(f"windows: train {len(train_ds)}, test {len(test_ds)}")
    if args.balance_speed:
        train_ds = balance_by_speed(train_ds)

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
        "sitl_included": not args.no_sitl,
        "attitude_source": args.attitude,
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "train_windows": len(train_ds), "test_windows": len(test_ds),
        "best_test_rmse": history["best_test_rmse"],
        "elapsed_sec": history["elapsed_sec"],
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
