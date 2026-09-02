#!/usr/bin/env python
"""B2: replace the per-flight max normalisation of the motor channel with a fixed scale.

mario/data.py:83 does ``motor /= max|motor|`` over the whole flight. Two problems:

  * it is a leak -- at any instant during a flight, that flight's maximum is future
    information, and nothing online could reproduce it;
  * it destroys the absolute thrust level. Blackbird's channel is mass-normalised
    collective thrust, ~-10.7 m/s^2 with a std of 0.5-1.7. Dividing each flight by its own
    maximum pins every flight's mean near -0.86, erasing the between-flight differences
    (mass, aggressiveness) that the level encodes.

This wraps the loader instead of editing mario/ (work rule 3): the normalised channel is
multiplied back by the per-flight maximum read from thrust_data.csv, then divided by a
single fixed constant. Hover then sits at -1.0 in every flight and stays comparable across
flights and, later, across vehicles.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mario.config import Config  # noqa: E402
from mario.data import load_eval_sequences, load_split  # noqa: E402
from mario.dataset import BlackbirdDispDataset  # noqa: E402
from mario.evaluate import evaluate_split  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402
from mario.train import train  # noqa: E402
from mario.utils import set_seed  # noqa: E402

G = 9.80665   # hover thrust; makes the channel read -1.0 at hover for any vehicle


def rescale_motor(seq: dict, flight_dir: Path, scale: float) -> str:
    """Undo the per-flight max normalisation, then apply one fixed physical scale."""
    tp = flight_dir / "thrust_data.csv"
    if not tp.exists():
        return "no thrust_data.csv -- left as is"
    raw = np.loadtxt(tp, delimiter=",")
    peak = np.max(np.abs(raw[:, 1:4]), axis=0)          # exactly what data.py divided by
    m = seq["motor"].numpy().copy()
    m = m * peak[None, :]                                # back to m/s^2
    m = m / scale
    seq["motor"] = torch.tensor(m, dtype=torch.float32)
    return f"peak={np.round(peak, 3).tolist()} -> mean {m.mean(0).round(3).tolist()}"


def build(seqs, tag, cfg, step, key_extra):
    key = hashlib.md5(f"{tag}|{step}|{key_extra}".encode()).hexdigest()[:12]
    cache = ROOT / "mario_sitl" / "results" / f".bbwin_{tag}_{key}.pt"
    ds = BlackbirdDispDataset([], cfg.data.window_size, step,
                              cfg.data.label_start_index, cfg.data.label_stride)
    if cache.exists():
        ds.windows = torch.load(cache, weights_only=False)
        print(f"  {tag}: {len(ds)} windows (cache)")
        return ds
    ds = BlackbirdDispDataset(seqs, cfg.data.window_size, step,
                              cfg.data.label_start_index, cfg.data.label_stride)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds.windows, cache)
    print(f"  {tag}: {len(ds)} windows (built, cached)")
    return ds


def evaluate(net, cfg, device, data_root, scale, fixed):
    """Use the repo's own protocol verbatim so the numbers compare to runs/*/results.json.

    Note that load_eval_sequences reads BOTH seen and unseen from the eval/ split -- the
    published seen figures are eval/clover etc., not test/. Reimplementing this was how a
    first version produced numbers that looked comparable and were not.
    """
    seen_seqs, unseen_seqs = load_eval_sequences(cfg.data, verbose=False)
    if fixed:
        by_name = {t.split("/")[0]: t for t in list(cfg.data.seen) + list(cfg.data.unseen)}
        for pairs in (seen_seqs, unseen_seqs):
            for name, seq in pairs:
                d = Path(data_root) / "eval" / by_name[name]
                rescale_motor(seq, d, scale)
    seen = evaluate_split(net, seen_seqs, device, cfg.data, title="SEEN")
    unseen = evaluate_split(net, unseen_seqs, device, cfg.data, title="UNSEEN")
    return {"seen": seen, "unseen": unseen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "trial8.yaml"))
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bb_fixedscale")
    ap.add_argument("--scale", type=float, default=G)
    ap.add_argument("--uncertainty-weight", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="keep mario/data.py's per-flight normalisation (control run)")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    cfg.data.data_dir = str(Path(cfg.data.data_dir).expanduser())
    if a.epochs:
        cfg.train.epochs = a.epochs
    if a.uncertainty_weight is not None:
        cfg.train.uncertainty_weight = a.uncertainty_weight
    if a.lr is not None:
        cfg.train.lr = a.lr
    if a.weight_decay is not None:
        cfg.train.weight_decay = a.weight_decay
    cfg.output_dir = str(a.out)
    set_seed(cfg.seed)
    device = torch.device("cuda")
    root = Path(cfg.data.data_dir)
    fixed = not a.baseline
    print(f"motor channel: {'FIXED scale /%.4f' % a.scale if fixed else 'per-flight max (baseline)'}")

    seqs = {}
    for split, trajs in (("train", cfg.data.seen), ("test", cfg.data.seen)):
        acc = []
        for traj in trajs:
            d = root / split / traj
            if not d.exists():
                continue
            pairs = load_split(root, [traj], split, dt=cfg.data.dt, verbose=False)
            if not pairs:
                continue
            name, seq = pairs[0]
            if fixed:
                print(f"  {split}/{name}: {rescale_motor(seq, d, a.scale)}")
            acc.append(seq)
        seqs[split] = acc

    tagsuffix = f"fixed{a.scale:.4f}" if fixed else "perflight"
    train_ds = build(seqs["train"], f"train_{tagsuffix}", cfg, cfg.data.train_step_size, tagsuffix)
    test_ds = build(seqs["test"], f"test_{tagsuffix}", cfg, cfg.data.test_step_size, tagsuffix)

    net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                             expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(device)
    t0 = time.time()
    hist = train(net, train_ds, test_ds, cfg, device)
    net.load_state_dict(torch.load(Path(cfg.output_dir) / "best.pt",
                                   map_location=device, weights_only=True))
    net.eval()
    res = evaluate(net, cfg, device, root, a.scale, fixed)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=2))
    (a.out / "run_meta.json").write_text(json.dumps({
        "motor_normalisation": "fixed" if fixed else "per_flight_max",
        "scale": a.scale, "uncertainty_weight": cfg.train.uncertainty_weight,
        "epochs": cfg.train.epochs, "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "best_test_rmse": hist["best_test_rmse"],
        "elapsed_sec": time.time() - t0,
    }, indent=2))
    for k, v in res.items():
        print(f"{k.upper():>7}  ATE {v['mean_ate']:.4f}  TDE {v['mean_tde']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
