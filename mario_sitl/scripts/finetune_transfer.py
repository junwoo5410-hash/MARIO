#!/usr/bin/env python
"""Fine-tune a Blackbird-trained MARIO onto another dataset, four ways.

The point of the study is not one number but the comparison: how much of the network
has to move before it works on a new airframe and flight regime. The arms differ only
in which parameters are allowed to change.

  head     disp/cov decoders only -- the linear probe. If this recovers most of the
           gap, the encoder features transferred and only the output mapping was wrong.
  trunk    the three CNN encoders are frozen, everything after them trains. Tests
           whether the raw-IMU front end is reusable while the dynamics model is not.
  full     everything trains, from the Blackbird weights.
  scratch  everything trains, from a random init, on target data only. The control
           that says whether pre-training bought anything at all.

Frozen submodules are also held in eval mode for the whole run, so their BatchNorm
running statistics do not quietly adapt -- otherwise "frozen" would be a lie and the
head arm would be doing feature adaptation it is not credited with.

Checkpoint selection uses a validation split taken as the tail of each training
sequence, never the official test sequences. Blackbird taught us that selecting on
something that shares flights with training barely sees the systematic bias, so the
final numbers here are always the trajectory-level rollout on held-out sequences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config                      # noqa: E402
from mario.dataset import BlackbirdDispDataset       # noqa: E402
from mario.evaluate import evaluate_split            # noqa: E402
from mario.model import CausalMambaDispNet           # noqa: E402
from mario.dataset import collate_fn                 # noqa: E402
from mario.losses import huber_loss, uncertainty_loss  # noqa: E402
from mario.train import train, _forward_batch         # noqa: E402
from bimamba_model import BiMambaDispNet             # noqa: E402

#: fraction of each training sequence kept back for checkpoint selection
VAL_TAIL = 0.2
#: samples dropped between the training part and the validation tail, so no window
#: can straddle the boundary and appear in both
VAL_GAP = 1200


def freeze(net: torch.nn.Module, mode: str) -> List[torch.nn.Module]:
    """Disable gradients for the parts ``mode`` holds fixed; return the frozen modules."""
    if mode in ("full", "scratch"):
        return []
    if mode == "head":
        keep = ("disp_decoder", "cov_decoder")
        frozen = [m for n, m in net.named_children() if n not in keep]
    elif mode == "trunk":
        frozen = [net.imu_encoder, net.ori_encoder, net.motor_encoder]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    for module in frozen:
        for p in module.parameters():
            p.requires_grad_(False)
    return frozen


def pin_eval(modules: List[torch.nn.Module]) -> None:
    """Make ``.train()`` a no-op on these modules so their BatchNorm stays frozen."""
    for module in modules:
        module.eval()
        module.train = lambda self_mode=True, _m=module: _m  # type: ignore[assignment]


def split_tail(seq: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
    """Split one sequence into (head part for training, tail part for validation)."""
    n = len(seq["acc"])
    cut = int(n * (1.0 - VAL_TAIL))
    head = {k: v[: max(cut - VAL_GAP, 0)] for k, v in seq.items()}
    tail = {k: v[cut:] for k, v in seq.items()}
    return head, tail


def load_target(dataset: str, align: str, yaw: float, n_train: int | None,
                verbose: bool = True):
    """Return (train sequences, test (name, sequence) pairs) for the target dataset."""
    if dataset == "euroc":
        import euroc_data
        lists = euroc_data.read_lists()
        names = lists["train"] if n_train is None else lists["train"][:n_train]
        train = [s for _, s in euroc_data.load_split(names, dt=0.01, align=align,
                                                     yaw_deg=yaw, verbose=verbose)]
        test = euroc_data.load_split(lists["test"], dt=0.01, align=align,
                                     yaw_deg=yaw, verbose=verbose)
        return train, test, names
    raise ValueError(dataset)


#: window construction rotates every label through pypose and takes minutes, so the
#: built windows are cached; the key carries everything that changes their content
CACHE_DIR = ROOT / "mario_sitl" / "results" / "transfer" / "cache"


def build_windows(seqs, cfg, step: int, key: str) -> BlackbirdDispDataset:
    d = cfg.data
    digest = hashlib.md5(f"{key}|{step}".encode()).hexdigest()[:12]
    path = CACHE_DIR / f"win_{digest}.pt"
    ds = BlackbirdDispDataset([], d.window_size, step, d.label_start_index, d.label_stride)
    if path.exists():
        ds.windows = torch.load(path, weights_only=False)
        print(f"  windows {key}: {len(ds)} (cache)")
        return ds
    ds = BlackbirdDispDataset(seqs, d.window_size, step, d.label_start_index, d.label_stride)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ds.windows, path)
    print(f"  windows {key}: {len(ds)} (built)")
    return ds


def train_select_ate(net, train_ds, val_pairs, cfg, device):
    """Same optimisation as mario.train.train, but checkpointed on rollout ATE.

    mario.train.train selects on window RMSE, which is the error of one 0.09 s
    displacement. Blackbird already showed that a model can improve that while its
    integrated trajectory gets worse -- a small constant bias is nearly invisible per
    window and dominates once chained. Transfer makes that worse, not better: at 100
    epochs the window RMSE kept falling while test ATE rose above the zero-shot value.
    So selection here uses the metric the model is actually judged on, computed by
    rolling out the held-back tail of each training sequence.
    """
    import torch.utils.data as Data
    t = cfg.train
    loader = Data.DataLoader(train_ds, batch_size=t.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=t.num_workers)
    opt = torch.optim.Adam(net.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=t.scheduler_factor, patience=t.scheduler_patience,
        min_lr=t.scheduler_min_lr)

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = float("inf"), 0, []
    started = time.time()
    for epoch in range(1, t.epochs + 1):
        net.train()
        total, nb = 0.0, 0
        for batch in loader:
            pred, cov, gt = _forward_batch(net, batch, device)
            res = pred - gt
            loss = t.loss_weight * huber_loss(res, delta=t.huber_delta)
            loss = loss + t.uncertainty_weight * uncertainty_loss(res.detach(), cov)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip)
            opt.step()
            total += loss.item()
            nb += 1

        val = evaluate_split(net, val_pairs, device, cfg.data, verbose=False)
        ate = val["mean_ate"]
        sched.step(ate)
        if np.isfinite(ate) and ate < best:
            best, best_epoch = ate, epoch
            torch.save(net.state_dict(), cfg.checkpoint_path)
        history.append({"epoch": epoch, "train_loss": total / max(nb, 1), "val_ate": ate})
        if epoch == 1 or epoch % t.log_every == 0 or epoch == t.epochs:
            print(f"Epoch {epoch:3d}/{t.epochs} | loss {history[-1]['train_loss']:.5f} | "
                  f"val ATE {ate:.4f} | best {best:.4f} @{best_epoch} | "
                  f"{time.time() - started:.0f}s")
    print(f"\nbest val ATE {best:.4f} at epoch {best_epoch}")
    return {"best_val_ate": best, "best_epoch": best_epoch,
            "elapsed_sec": time.time() - started, "history": history}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("head", "trunk", "full", "scratch"))
    ap.add_argument("--dataset", default="euroc", choices=("euroc",))
    ap.add_argument("--init", type=Path, default=ROOT / "runs" / "m_s42" / "best.pt",
                    help="source checkpoint; ignored when --mode scratch")
    ap.add_argument("--arch", default="causal", choices=("causal", "bi"))
    ap.add_argument("--align", default="gravity", choices=("none", "gravity"))
    ap.add_argument("--yaw-deg", type=float, default=None,
                    help="default 185, picked on non-test data in the zero-shot sweep")

    ap.add_argument("--n-train", type=int, default=None,
                    help="use only the first N training sequences (data-efficiency curve)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: the source lr for scratch, a tenth of it otherwise")
    ap.add_argument("--select", default="ate", choices=("ate", "rmse"),
                    help="checkpoint selection metric: rollout ATE on the held-back "
                         "tails (default) or per-window RMSE, which is what "
                         "mario.train.train uses")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=str(ROOT / "configs" / "trial8.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.yaw_deg is None:
        a.yaw_deg = 185.0

    cfg = Config.load(a.config)
    cfg.seed = a.seed
    cfg.train.epochs = a.epochs
    cfg.train.lr = a.lr if a.lr is not None else (
        cfg.train.lr if a.mode == "scratch" else cfg.train.lr / 10.0)
    cfg.output_dir = str(a.out)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"mode={a.mode}  dataset={a.dataset}  seed={a.seed}  lr={cfg.train.lr:.3g}  "
          f"epochs={a.epochs}  align={a.align} yaw={a.yaw_deg:g}")
    train_seqs, test_pairs, train_names = load_target(
        a.dataset, a.align, a.yaw_deg, a.n_train)
    print(f"train sequences ({len(train_seqs)}): {' '.join(train_names)}")

    heads, tails = zip(*(split_tail(s) for s in train_seqs))
    d = cfg.data
    base = f"{a.dataset}|{a.align}|{a.yaw_deg}|{'-'.join(train_names)}"
    train_ds = build_windows(heads, cfg, d.train_step_size, base + "|train")
    val_ds = build_windows(tails, cfg, d.test_step_size, base + "|val")
    val_pairs = [(f"{n}_tail", t) for n, t in zip(train_names, tails)]
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("not enough data to train")
        return 1

    Net = BiMambaDispNet if a.arch == "bi" else CausalMambaDispNet
    net = Net(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
              expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(device)
    if a.mode != "scratch":
        net.load_state_dict(torch.load(a.init, map_location=device, weights_only=True))
    pin_eval(freeze(net, a.mode))
    n_train_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in net.parameters())
    print(f"trainable: {n_train_p:,} / {n_all:,} ({100 * n_train_p / n_all:.1f}%)")

    t0 = time.time()
    if a.select == "ate":
        hist = train_select_ate(net, train_ds, val_pairs, cfg, device)
    else:
        hist = train(net, train_ds, val_ds, cfg, device)
    net.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device,
                                   weights_only=True))
    net.eval()
    res = evaluate_split(net, test_pairs, device, cfg.data,
                         title=f"{a.mode.upper()}  {a.dataset} test")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps({
        "mode": a.mode, "dataset": a.dataset, "seed": a.seed, "lr": cfg.train.lr,
        "epochs": a.epochs, "align": a.align, "yaw_deg": a.yaw_deg,
        "n_train_seqs": len(train_seqs), "train_names": train_names,
        "trainable_params": n_train_p, "total_params": n_all,
        "select": a.select,
        "best_val_rmse": hist.get("best_test_rmse"),
        "best_val_ate": hist.get("best_val_ate"),
        "best_epoch": hist.get("best_epoch"),
        "elapsed_sec": time.time() - t0,
        "result": res,
    }, indent=2))
    print(f"\n{a.mode.upper():>8}  ATE {res['mean_ate']:.4f}  TDE {res['mean_tde']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
