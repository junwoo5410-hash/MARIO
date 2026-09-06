#!/usr/bin/env python
"""Roll a checkpoint out over the WHOLE trajectory, so the numbers compare to AirIO's.

mario.evaluate.rollout_trajectory steps its window by ``window_size``. Inside a window
the labels run from index s+14 to s+1013, so the next window's first label sits at
s+1014 and the chain test ``if t_start in positions`` fails: every window after the
first is silently dropped. On the Blackbird eval split that leaves 113 poses, about
10.2 s, covering 46-71 % of each trajectory. Every MARIO number in this repository was
produced that way, so they compare to each other -- but AirIO integrates the entire
sequence from one ground-truth anchor (evaluate_motion.py calls integrate() with
gtinit=False, save_full_traj=True), so MARIO-vs-AirIO was not measured over the same
horizon, and a shorter horizon flatters whoever gets it.

Stepping the window by 999 instead makes the last label of one window land exactly on
the first label of the next (s+1013 = (s+999)+14), so the chain is continuous and the
rollout spans the sequence. Inference is unchanged: same per-window forward, state
reset at each window, anchored once at the first ground-truth position.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config, DEFAULT_SEEN, DEFAULT_UNSEEN   # noqa: E402
from mario.data import load_split                               # noqa: E402
from mario.model import CausalMambaDispNet                      # noqa: E402
from bimamba_model import BiMambaDispNet                        # noqa: E402


@torch.no_grad()
def rollout_full(net, data, device, window_size=1000, label_start_index=14,
                 label_stride=9) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    net.eval()
    gt_rot, gt_pos = data["gt_orientation"], data["gt_translation"]
    seq_len = len(data["acc"])
    # chosen so consecutive windows produce label indices that meet exactly
    step = window_size - label_start_index + label_stride - label_stride
    step = window_size - 1  # = s+1013 -> (s+999)+14 for the default geometry

    steps: List[Tuple[int, int, torch.Tensor]] = []
    start = 0
    while start + window_size <= seq_len:
        sl = slice(start, start + window_size)
        pred, _ = net(data["acc"][sl].unsqueeze(0).to(device),
                      data["gyro"][sl].unsqueeze(0).to(device),
                      gt_rot[sl].unsqueeze(0).to(device).Log().tensor().float())
        pred = pred.squeeze(0).cpu()
        for i in range(pred.shape[0]):
            t = start + label_start_index + i * label_stride
            t_next = t + label_stride
            if t_next < seq_len:
                steps.append((t, t_next, gt_rot[t] @ pred[i]))
        start += step

    if len(steps) < 2:
        return None
    steps.sort(key=lambda s: s[0])
    positions = {steps[0][0]: gt_pos[steps[0][0]].clone()}
    for t, t_next, disp in steps:
        if t in positions:
            positions[t_next] = positions[t] + disp
    ts = sorted(positions)
    if len(ts) < 10:
        return None
    return gt_pos[ts].numpy(), torch.stack([positions[t] for t in ts]).numpy()


def metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = np.linalg.norm(pred - gt, axis=1)
    ate = float(np.sqrt(np.mean(err ** 2)))
    dist = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    return {"ATE": ate, "TDE": ate / dist * 100.0 if dist > 0 else 0.0,
            "distance": dist, "n_poses": int(len(gt))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/src/gs25122/blackbird_data")
    ap.add_argument("--seeds", default="42 1 2 3")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "mario_sitl" / "results" / "full_rollout.json")
    a = ap.parse_args()

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in a.seeds.split()]

    seqs = {"seen": [], "unseen": []}
    for split, trajs in (("seen", DEFAULT_SEEN), ("unseen", DEFAULT_UNSEEN)):
        for t in trajs:
            name, seq = load_split(a.data_root, [t], "eval", dt=cfg.data.dt,
                                   verbose=False)[0]
            seqs[split].append((name, seq))

    out: Dict[str, dict] = {}
    variants = [("NM", CausalMambaDispNet, "nm"), ("MB", BiMambaDispNet, "mb")]
    keep = []
    for tag, Net, prefix in variants:
        ck = ROOT / "runs" / f"{prefix}_s{seeds[0]}" / "best.pt"
        if not ck.exists():
            continue
        try:                       # motor-era checkpoints no longer fit these classes
            Net(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                expand=cfg.model.expand, num_layers=cfg.model.num_layers).load_state_dict(
                    torch.load(ck, map_location="cpu", weights_only=True))
        except RuntimeError as exc:
            print(f"[skip] {tag}: {str(exc).splitlines()[0]}")
            continue
        keep.append((tag, Net, prefix))
    variants = keep
    for tag, Net, prefix in variants:
        out[tag] = {}
        for split in ("seen", "unseen"):
            per_seed = []
            for sd in seeds:
                net = Net(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                          expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(dev)
                net.load_state_dict(torch.load(ROOT / "runs" / f"{prefix}_s{sd}" / "best.pt",
                                               map_location=dev, weights_only=True))
                per_traj = {}
                for name, seq in seqs[split]:
                    r = rollout_full(net, seq, dev, cfg.data.window_size,
                                     cfg.data.label_start_index, cfg.data.label_stride)
                    if r is not None:
                        per_traj[name] = metrics(*r)
                per_seed.append({
                    "seed": sd,
                    "mean_ate": float(np.mean([m["ATE"] for m in per_traj.values()])),
                    "mean_tde": float(np.mean([m["TDE"] for m in per_traj.values()])),
                    "per_trajectory": per_traj,
                })
            ates = np.array([p["mean_ate"] for p in per_seed])
            tdes = np.array([p["mean_tde"] for p in per_seed])
            out[tag][split] = {"ate_mean": float(ates.mean()), "ate_std": float(ates.std()),
                               "tde_mean": float(tdes.mean()), "tde_std": float(tdes.std()),
                               "seeds": per_seed}
            print(f"{tag:<3} {split:<7} ATE {ates.mean():.4f} +-{ates.std():.4f}   "
                  f"TDE {tdes.mean():.4f} +-{tdes.std():.4f}")

    # distances, so AirIO's published ATE can be turned into a TDE on the same span
    out["distances"] = {split: {n: float(np.linalg.norm(
        np.diff(s["gt_translation"].numpy(), axis=0), axis=1).sum())
        for n, s in seqs[split]} for split in seqs}
    a.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
