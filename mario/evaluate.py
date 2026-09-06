"""Trajectory-level rollout and metrics (ATE / TDE)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mario.data import Sequence

#: a rollout needs at least this many integrated poses to be worth scoring
MIN_ROLLOUT_POSES = 10


@torch.no_grad()
def rollout_trajectory(
    net: torch.nn.Module,
    data: Sequence,
    device: torch.device,
    window_size: int = 1000,
    label_start_index: int = 14,
    label_stride: int = 9,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Predict displacements window by window and integrate them into a trajectory.

    Displacements are rotated back to the world frame with the ground-truth attitude,
    then chained from the first ground-truth position. Returns ``(gt_xyz, pred_xyz)``
    sampled at the predicted timestamps, or ``None`` if the sequence is too short.
    """
    net.eval()
    gt_rot, gt_pos = data["gt_orientation"], data["gt_translation"]
    seq_len = len(data["acc"])

    steps: List[Tuple[int, int, torch.Tensor]] = []
    for start in range(0, seq_len - window_size, window_size):
        end = start + window_size
        acc = data["acc"][start:end].unsqueeze(0).to(device)
        gyro = data["gyro"][start:end].unsqueeze(0).to(device)
        rot_so3 = gt_rot[start:end].unsqueeze(0).to(device).Log().tensor().float()

        pred_d, _ = net(acc, gyro, rot_so3)
        pred_d = pred_d.squeeze(0).cpu()

        for i in range(pred_d.shape[0]):
            t_idx = start + label_start_index + i * label_stride
            t_next = t_idx + label_stride
            if t_next < seq_len:
                steps.append((t_idx, t_next, gt_rot[t_idx] @ pred_d[i]))

    if len(steps) < 2:
        return None

    steps.sort(key=lambda s: s[0])
    positions = {steps[0][0]: gt_pos[steps[0][0]].clone()}
    for t_start, t_next, disp in steps:
        if t_start in positions:
            positions[t_next] = positions[t_start] + disp

    timestamps = sorted(positions)
    if len(timestamps) < MIN_ROLLOUT_POSES:
        return None

    pred_arr = torch.stack([positions[t] for t in timestamps]).numpy()
    return gt_pos[timestamps].numpy(), pred_arr


def trajectory_metrics(gt_arr: np.ndarray, pred_arr: np.ndarray) -> Optional[Dict[str, float]]:
    """Absolute trajectory error and it as a percentage of the distance travelled."""
    errors = np.linalg.norm(pred_arr - gt_arr, axis=1)
    ate = float(np.sqrt(np.mean(errors**2)))
    if not np.isfinite(ate):
        return None
    distance = float(np.sum(np.linalg.norm(np.diff(gt_arr, axis=0), axis=1)))
    return {
        "ATE": ate,
        "TDE": (ate / distance * 100.0) if distance > 0 else 0.0,
        "distance": distance,
    }


def evaluate_trajectory(net, data, device, cfg) -> Optional[Dict[str, float]]:
    """Rollout + metrics for a single sequence, using a :class:`DataConfig`."""
    rollout = rollout_trajectory(
        net,
        data,
        device,
        window_size=cfg.window_size,
        label_start_index=cfg.label_start_index,
        label_stride=cfg.label_stride,
    )
    if rollout is None:
        return None
    return trajectory_metrics(*rollout)


def evaluate_split(
    net,
    sequences: List[Tuple[str, Sequence]],
    device,
    cfg,
    title: str = "",
    verbose: bool = True,
) -> Dict[str, object]:
    """Evaluate every sequence in a split and return per-trajectory plus mean metrics."""
    if verbose and title:
        print(f"\n{'=' * 46}\n  {title}\n{'=' * 46}")
        print(f"{'trajectory':<18}{'ATE [m]':>10}{'TDE [%]':>10}")
        print("-" * 38)

    per_trajectory: Dict[str, Dict[str, float]] = {}
    for name, data in sequences:
        metrics = evaluate_trajectory(net, data, device, cfg)
        if metrics is None:
            if verbose:
                print(f"{name:<18}{'n/a':>10}{'n/a':>10}")
            continue
        per_trajectory[name] = metrics
        if verbose:
            print(f"{name:<18}{metrics['ATE']:>10.3f}{metrics['TDE']:>10.2f}")

    ate_values = [m["ATE"] for m in per_trajectory.values()]
    mean_ate = float(np.mean(ate_values)) if ate_values else float("nan")
    mean_tde = (
        float(np.mean([m["TDE"] for m in per_trajectory.values()]))
        if per_trajectory
        else float("nan")
    )
    if verbose:
        print("-" * 38)
        print(f"{'mean':<18}{mean_ate:>10.3f}{mean_tde:>10.2f}")

    return {"mean_ate": mean_ate, "mean_tde": mean_tde, "per_trajectory": per_trajectory}
