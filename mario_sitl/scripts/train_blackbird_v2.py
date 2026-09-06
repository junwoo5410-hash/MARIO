#!/usr/bin/env python
"""Training driver for the Blackbird experiments.

Wraps mario.train with the knobs the sweeps needed -- architecture, seed, capacity,
Huber delta, held-out validation trajectory, rotation augmentation -- and caches the
built windows, which are expensive because every label goes through pypose.

The motor/thrust channel this script used to rescale is gone: it was a per-flight DC
level the network read as a flight identifier, and removing it cut unseen ATE by 79 %
over the full trajectory. See mario_sitl/BLACKBIRD_IMPROVEMENTS.md.
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
import pypose as pp
import torch
import torch.utils.data as Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mario.config import Config  # noqa: E402
from mario.data import load_eval_sequences, load_split  # noqa: E402
from mario.dataset import BlackbirdDispDataset  # noqa: E402
from mario.evaluate import evaluate_split  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402
from bimamba_model import BiMambaDispNet  # noqa: E402
from mario.train import train  # noqa: E402
from mario.utils import set_seed  # noqa: E402

G = 9.80665   # hover thrust; makes the channel read -1.0 at hover for any vehicle


class RotAugDataset(Data.Dataset):
    """Random rotation of the body frame about its own z axis, drawn per window.

    Blackbird's thrust channel is purely along body z -- checked over all 20
    thrust_data.csv files, ch0 and ch1 are identically zero and ch2 carries the
    mass-normalised collective -- so a rotation about z leaves the motor input
    untouched while acc, gyro and the body-frame displacement label all rotate with
    the frame. The map is an exact symmetry of the task, not an approximation: it is
    the data a physically yaw-rotated IMU mount would have produced.

    Why it should help generalisation: every Blackbird flight is flown ``yawForward``,
    so heading is tied to path direction and the network can key on absolute in-plane
    direction to tell the five training trajectories apart. Randomising the mounting
    yaw removes that shortcut without touching the physics.

    With C = Rz(psi) mapping old body coordinates to new ones,
        acc, gyro, label  ->  C v      (row vectors: ``v @ C.T``)
        R_wb              ->  R_wb C^T (so ``R_wb.Inv() @ world_disp`` gives C d_b)
    """

    def __init__(self, base: Data.Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        w = self.base[idx]
        psi = float(torch.rand(()) * 2.0 * math.pi - math.pi)
        c, s = math.cos(psi), math.sin(psi)
        Ct = torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        Cinv = pp.SO3(torch.tensor([0.0, 0.0, math.sin(-psi / 2.0), math.cos(-psi / 2.0)]))
        return {
            "acc": w["acc"] @ Ct,
            "gyro": w["gyro"] @ Ct,
            "gt_rot": w["gt_rot"] @ Cinv,
            "gt_disp": w["gt_disp"] @ Ct,
        }


def load_seqs(root, split, trajs, verbose=True):
    """Load one split for a list of trajectories, skipping any that are absent."""
    acc = []
    for traj in trajs:
        d = root / split / traj
        if not d.exists():
            continue
        pairs = load_split(root, [traj], split, dt=0.01, verbose=False)
        if not pairs:
            continue
        name, seq = pairs[0]
        acc.append(seq)
    return acc


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


def evaluate(net, cfg, device):
    """Use the repo's own protocol verbatim so the numbers compare to runs/*/results.json.

    Note that load_eval_sequences reads BOTH seen and unseen from the eval/ split -- the
    published seen figures are eval/clover etc., not test/. Reimplementing this was how a
    first version produced numbers that looked comparable and were not.
    """
    seen_seqs, unseen_seqs = load_eval_sequences(cfg.data, verbose=False)
    seen = evaluate_split(net, seen_seqs, device, cfg.data, title="SEEN")
    unseen = evaluate_split(net, unseen_seqs, device, cfg.data, title="UNSEEN")
    return {"seen": seen, "unseen": unseen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "trial8.yaml"))
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bb_fixedscale")
    ap.add_argument("--uncertainty-weight", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--arch", default="causal", choices=("causal", "bi"),
                    help="bi runs the Mamba stack in both directions -- not deployable, "
                         "it is the controlled test of what AirIO's bidirectional GRU buys")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed replication -- B1 looked like a win on seed 42 and did not "
                         "hold on 1 and 3, so candidates are checked across seeds now")

    ap.add_argument("--huber-delta", type=float, default=None,
                    help="MARIO uses 0.002, AirIO 0.05. At MARIO's residual scale 0.002 puts "
                         "everything in Huber's linear arm, so it trains an L1 objective "
                         "while ATE scores RMS")
    ap.add_argument("--loss-weight", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=None,
                    help="Mamba trunk width (default 64). AirIO carries 206k params "
                         "unidirectional and 387k bidirectional against MARIO's 96k/132k, "
                         "and holding a trajectory out made things worse, so the model "
                         "looks capacity-limited rather than overfit")
    ap.add_argument("--expand", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--val-traj", default=None,
                    help="hold this seen trajectory out of training and select best.pt on "
                         "it instead of on the seen test split. mario/data.py:134 builds "
                         "the selection set from cfg.seen, so checkpoint selection is "
                         "otherwise 100%% seen-fit and picks the most overfit epoch")
    ap.add_argument("--rot-aug", action="store_true",
                    help="random body-z rotation of each training window (exact symmetry)")

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
    if a.huber_delta is not None:
        cfg.train.huber_delta = a.huber_delta
    if a.loss_weight is not None:
        cfg.train.loss_weight = a.loss_weight
    cfg.output_dir = str(a.out)
    if a.seed is not None:
        cfg.seed = a.seed
    set_seed(cfg.seed)
    device = torch.device("cuda")
    root = Path(cfg.data.data_dir)

    train_trajs = list(cfg.data.seen)
    held: list[str] = []
    if a.val_traj:
        held = [t for t in train_trajs if t.split("/")[0] == a.val_traj]
        if not held:
            print(f"--val-traj {a.val_traj!r} is not in cfg.data.seen: "
                  f"{[t.split('/')[0] for t in train_trajs]}")
            return 2
        train_trajs = [t for t in train_trajs if t.split("/")[0] != a.val_traj]
        print(f"validation trajectory: {a.val_traj} (held out of training)")

    tagsuffix = "nomotor"
    # the cache key must carry the trajectory list, or a held-out run would silently
    # reuse the full-seen window cache
    tr_key = f"{tagsuffix}|{','.join(sorted(t.split('/')[0] for t in train_trajs))}"
    train_seqs = load_seqs(root, "train", train_trajs)
    train_ds = build(train_seqs, f"train_{tagsuffix}", cfg, cfg.data.train_step_size, tr_key)

    if held:
        # both splits of the held-out flight, so selection sees as much of it as possible
        val_seqs = load_seqs(root, "train", held)
        val_seqs += load_seqs(root, "test", held)
        val_key = f"{tagsuffix}|val:{a.val_traj}"
        test_ds = build(val_seqs, f"val_{tagsuffix}", cfg, cfg.data.test_step_size, val_key)
    else:
        val_seqs = load_seqs(root, "test", cfg.data.seen)
        test_ds = build(val_seqs, f"test_{tagsuffix}", cfg, cfg.data.test_step_size, tagsuffix)

    if a.rot_aug:
        train_ds = RotAugDataset(train_ds)
        print(f"rot-aug: on ({len(train_ds)} training windows, random yaw per window)")

    Net = BiMambaDispNet if a.arch == "bi" else CausalMambaDispNet
    kw = dict(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
              expand=a.expand if a.expand is not None else cfg.model.expand,
              num_layers=a.num_layers if a.num_layers is not None else cfg.model.num_layers)
    if a.d_model is not None:
        kw["d_model"] = a.d_model
    net = Net(**kw).to(device)
    print(f"arch: {a.arch}  params: {sum(p.numel() for p in net.parameters()):,}")
    t0 = time.time()
    hist = train(net, train_ds, test_ds, cfg, device)
    net.load_state_dict(torch.load(Path(cfg.output_dir) / "best.pt",
                                   map_location=device, weights_only=True))
    net.eval()
    res = evaluate(net, cfg, device)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=2))
    (a.out / "run_meta.json").write_text(json.dumps({
        "arch": a.arch,
        "huber_delta": cfg.train.huber_delta, "loss_weight": cfg.train.loss_weight,
        "d_model": a.d_model, "expand": kw["expand"],
        "num_layers": kw["num_layers"],
        "n_params": sum(p.numel() for p in net.parameters()),
        "val_traj": a.val_traj,
        "rot_aug": bool(a.rot_aug),
        "uncertainty_weight": cfg.train.uncertainty_weight,
        "epochs": cfg.train.epochs, "lr": cfg.train.lr, "seed": cfg.seed,
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
