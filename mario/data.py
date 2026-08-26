"""Loading and time-alignment of raw Blackbird sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pypose as pp
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

Sequence = Dict[str, torch.Tensor]

# Blackbird ground truth is published in a NED-ish world frame with a rotated
# body frame; these bring it into the frame the IMU is expressed in.
R_W_NED = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
R_B_I = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

VELOCITY_SMOOTHING_WINDOW = 5


def _ground_truth_in_imu_frame(gt_raw: np.ndarray) -> np.ndarray:
    """Convert raw ground-truth poses to ``[t, xyz, qxyzw]`` in the IMU world frame."""
    rows = []
    for d in gt_raw:
        ts = d[0] / 1e6
        t_i = d[1:4]
        # raw quaternion is stored (w, x, y, z); scipy wants (x, y, z, w)
        R_i = Rotation.from_quat([d[5], d[6], d[7], d[4]]).as_matrix()
        R_it = R_W_NED @ R_i @ R_B_I
        t_it = R_W_NED @ t_i
        q_it = Rotation.from_matrix(R_it).as_quat()
        rows.append([ts, *t_it, *q_it])
    return np.array(rows)


def _finite_difference_velocity(data: np.ndarray) -> np.ndarray:
    """Smoothed world-frame velocity from ground-truth positions."""
    vel = np.diff(data[:, 1:4], axis=0) / np.diff(data[:, 0])[:, None]
    vel = np.concatenate([vel[:1], vel], axis=0)
    kernel = np.ones(VELOCITY_SMOOTHING_WINDOW) / VELOCITY_SMOOTHING_WINDOW
    return np.stack([np.convolve(vel[:, i], kernel, "same") for i in range(3)], axis=1)


def load_blackbird(data_path: str | Path, dt: float = 0.01) -> Sequence:
    """Load one Blackbird flight and resample every signal onto a uniform ``dt`` grid.

    ``thrust_data.csv`` is optional; when it is missing the motor channel is zeroed
    so the same network can still run on IMU-only sequences.
    """
    data_path = Path(data_path)
    imu_raw = np.loadtxt(data_path / "imu_data.csv", delimiter=",")
    gt_raw = np.loadtxt(data_path / "groundTruthPoses.csv", delimiter=",")

    thrust_path = data_path / "thrust_data.csv"
    has_thrust = thrust_path.exists()
    thrust_raw = np.loadtxt(thrust_path, delimiter=",") if has_thrust else None

    data = _ground_truth_in_imu_frame(gt_raw)
    gt_traj = np.concatenate([data, _finite_difference_velocity(data)], axis=1)

    new_times = np.arange(imu_raw[0, 0], imu_raw[-1, 0] - dt - 0.001, dt)
    gyro = interp1d(imu_raw[:, 0], imu_raw[:, 1:4], axis=0)(new_times)
    accel = interp1d(imu_raw[:, 0], imu_raw[:, 4:7], axis=0)(new_times)

    # keep only the span covered by every source signal
    t_start = max(new_times[0], data[0, 0])
    t_end = min(new_times[-1], data[-1, 0])
    if has_thrust:
        motor_interp = interp1d(
            thrust_raw[:, 0], thrust_raw[:, 1:4], axis=0, fill_value="extrapolate"
        )(new_times)
        t_start = max(t_start, thrust_raw[0, 0])
        t_end = min(t_end, thrust_raw[-1, 0])

    mask = (new_times >= t_start) & (new_times <= t_end)
    times, gyro, accel = new_times[mask], gyro[mask], accel[mask]

    if has_thrust:
        motor = motor_interp[mask]
        motor = motor / (np.max(np.abs(motor), axis=0) + 1e-8)
    else:
        motor = np.zeros((len(times), 3), dtype=np.float32)

    pos = interp1d(gt_traj[:, 0], gt_traj[:, 1:4], axis=0)(times)
    ori_quat = Slerp(gt_traj[:, 0], Rotation.from_quat(gt_traj[:, 4:8]))(times).as_quat()
    vel = interp1d(gt_traj[:, 0], gt_traj[:, 8:11], axis=0)(times)

    return {
        "time": torch.tensor(times, dtype=torch.float64),
        "acc": torch.tensor(accel, dtype=torch.float32),
        "gyro": torch.tensor(gyro, dtype=torch.float32),
        "motor": torch.tensor(motor, dtype=torch.float32),
        "gt_translation": torch.tensor(pos, dtype=torch.float32),
        "gt_orientation": pp.SO3(torch.tensor(ori_quat, dtype=torch.float32)),
        "velocity": torch.tensor(vel, dtype=torch.float32),
    }


def load_split(
    data_root: str | Path,
    trajectories: List[str],
    split: str,
    dt: float = 0.01,
    verbose: bool = True,
) -> List[Tuple[str, Sequence]]:
    """Load ``trajectories`` from one split directory, skipping any that are absent.

    Returns ``(short_name, sequence)`` pairs so downstream code never has to assume
    that every requested trajectory was present on disk.
    """
    data_root = Path(data_root).expanduser()
    loaded: List[Tuple[str, Sequence]] = []
    for traj in trajectories:
        path = data_root / split / traj
        if not path.exists():
            if verbose:
                print(f"  [skip] {split}/{traj} (not found)")
            continue
        loaded.append((traj.split("/")[0], load_blackbird(path, dt=dt)))
        if verbose:
            print(f"  [ok]   {split}/{traj}")
    return loaded


def load_training_sequences(cfg, verbose: bool = True):
    """Load the ``train`` and ``test`` splits of the SEEN trajectories."""
    root = Path(cfg.data_dir).expanduser()
    if verbose:
        print(f"Loading training data from {root}")
    train = load_split(root, cfg.seen, "train", dt=cfg.dt, verbose=verbose)
    test = load_split(root, cfg.seen, "test", dt=cfg.dt, verbose=verbose)
    return train, test


def load_eval_sequences(cfg, verbose: bool = True):
    """Load the ``eval`` split for both the SEEN and the held-out UNSEEN trajectories."""
    root = Path(cfg.data_dir).expanduser()
    if verbose:
        print(f"Loading evaluation data from {root}")
    seen = load_split(root, cfg.seen, "eval", dt=cfg.dt, verbose=verbose)
    unseen = load_split(root, cfg.unseen, "eval", dt=cfg.dt, verbose=verbose)
    return seen, unseen
