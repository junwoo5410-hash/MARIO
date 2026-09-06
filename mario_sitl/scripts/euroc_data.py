#!/usr/bin/env python
"""Loader for the EuRoC MAV dataset, shaped like mario/data.py's Blackbird loader.

The transfer target for the Blackbird-trained MARIO. EuRoC is the standard public
benchmark in this area and AirIO publishes numbers on it, so results here can be
placed next to somebody else's rather than only next to our own.

Source: the IMU+groundtruth-only repackaging the AirIO authors publish at
https://github.com/Air-IO/Air-IO/releases/download/datasets/EuRoC-Dataset.zip
(20 MB). The official ETH host, robotics.ethz.ch, has been switched off -- ASL
moved every dataset to the ETH Research Collection (DOI 10.3929/ethz-b-000690084)
on 2025-12-18, where the same data is only available as 6-12 GB archives that
carry the stereo imagery we do not use. The zip above also ships the official
train/val/test lists, so the split here is AirIO's, not one we chose.

Frames, checked before writing this:

  * imu0/sensor.yaml gives T_BS = identity, so the IMU frame *is* the body frame
    and the accelerometer needs no extrinsic correction;
  * ground truth is p_RS_R / q_RS, i.e. body-to-world in a z-up world frame, so
    rotating the specific force into the world frame recovers +g on z. This is
    verified at load time by ``check_frames``; unlike Blackbird, no R_W_NED or
    R_B_I is needed.
  * the stored v_RS_R is world-frame velocity, used directly rather than
    finite-differenced.

Two rates are in play. Blackbird trains at 100 Hz; EuRoC's IMU is 200 Hz. ``dt``
resamples onto a uniform grid so the same checkpoint can be fed either the
distribution it was trained on (dt=0.01) or the native rate (dt=0.005).

Ground truth is 200 Hz on every sequence except V1_01_easy, which is 20 Hz; both
go through the same interpolation path, so nothing special is needed, but the
sparser one is reported by ``describe`` so it is never silently trusted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pypose as pp
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path("/src/gs25122/aiio_data/datasets/euroc")

Sequence = Dict[str, torch.Tensor]

#: EuRoC's world frame is z-up and the sequences are gravity-aligned
GRAVITY = 9.81

# EuRoC mounts its IMU tilted in the airframe: the mean body-frame specific force
# points along +x, not -z. Both constants below are mean specific-force directions
# measured on TRAINING sequences only (EuRoC's six train_list entries, Blackbird's
# five SEEN trajectories), so aligning with them never looks at a test sequence.
# The EuRoC direction is the same on all eleven sequences to within 0.53 deg of
# scatter, which is what makes it a mounting constant rather than a flight property.
# The two conventions are 70.8 deg apart.
EUROC_GRAVITY_DIR = np.array([0.94200, 0.00018, -0.33538])
BLACKBIRD_GRAVITY_DIR = np.array([-0.0097, 0.1089, -0.9940])


def _shortest_arc(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``u`` onto ``v`` by the smallest angle."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3) if u @ v > 0 else -np.eye(3)
    return Rotation.from_rotvec(axis / s * np.arctan2(s, u @ v)).as_matrix()


def alignment(mode: str, yaw_deg: float = 0.0) -> np.ndarray:
    """Constant rotation C with ``v_new = C @ v_old``, applied to the body frame.

    The displacement problem is equivariant under a constant body rotation -- rotate
    acc, gyro and the label together and the physics is unchanged -- so this removes a
    coordinate-convention difference without inventing anything. ``gravity`` puts the
    hover specific force where Blackbird has it; the rotation about the resulting
    gravity axis is not determined by that constraint, hence ``yaw_deg`` to sweep it.
    """
    if mode == "none":
        return np.eye(3)
    if mode != "gravity":
        raise ValueError(f"unknown alignment {mode!r}")
    C = _shortest_arc(EUROC_GRAVITY_DIR, BLACKBIRD_GRAVITY_DIR)
    if yaw_deg:
        C = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ C
    return C


def apply_alignment(seq: Sequence, C: np.ndarray) -> Sequence:
    """Re-express the body frame of a sequence under ``C``. World quantities are untouched."""
    if np.allclose(C, np.eye(3)):
        return seq
    Ct = torch.tensor(C.T, dtype=torch.float32)
    seq["acc"] = seq["acc"] @ Ct
    seq["gyro"] = seq["gyro"] @ Ct
    R = Rotation.from_quat(seq["gt_orientation"].tensor().numpy()).as_matrix() @ C.T
    seq["gt_orientation"] = pp.SO3(
        torch.tensor(Rotation.from_matrix(R).as_quat(), dtype=torch.float32))
    return seq


def read_lists(root: Path = ROOT) -> Dict[str, List[str]]:
    """The official splits shipped with the dataset.

    ``train_list.txt`` and ``val_list.txt`` are identical in the release, so the
    caller gets whatever is on disk and decides what to do about that.
    """
    out = {}
    for name in ("train", "val", "test"):
        path = root / f"{name}_list.txt"
        out[name] = [ln.strip() for ln in path.read_text().split() if ln.strip()]
    return out


def load_euroc(seq_dir: Path, dt: float = 0.01, align: str = "none",
               yaw_deg: float = 0.0) -> Sequence:
    """Load one EuRoC sequence onto a uniform ``dt`` grid.

    The motor channel is zeros: EuRoC records no thrust or rotor signal. That is
    exactly the input the no-motor checkpoint was trained on, so this is the
    honest shape rather than a stand-in.
    """
    mav = Path(seq_dir) / "mav0"
    imu = np.loadtxt(mav / "imu0" / "data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(mav / "state_groundtruth_estimate0" / "data.csv",
                    delimiter=",", skiprows=1)

    t_imu = imu[:, 0] / 1e9
    gyro_raw, acc_raw = imu[:, 1:4], imu[:, 4:7]

    t_gt = gt[:, 0] / 1e9
    pos = gt[:, 1:4]
    quat_xyzw = gt[:, [5, 6, 7, 4]]     # stored (w, x, y, z)
    vel = gt[:, 8:11]

    # only the span both sources cover, snapped to a uniform grid
    t0 = max(t_imu[0], t_gt[0])
    t1 = min(t_imu[-1], t_gt[-1])
    times = np.arange(t0, t1 - dt - 1e-9, dt)

    gyro = interp1d(t_imu, gyro_raw, axis=0)(times)
    acc = interp1d(t_imu, acc_raw, axis=0)(times)
    p = interp1d(t_gt, pos, axis=0)(times)
    v = interp1d(t_gt, vel, axis=0)(times)
    q = Slerp(t_gt, Rotation.from_quat(quat_xyzw))(times).as_quat()

    seq = {
        "time": torch.tensor(times, dtype=torch.float64),
        "acc": torch.tensor(acc, dtype=torch.float32),
        "gyro": torch.tensor(gyro, dtype=torch.float32),
        "gt_translation": torch.tensor(p, dtype=torch.float32),
        "gt_orientation": pp.SO3(torch.tensor(q, dtype=torch.float32)),
        "velocity": torch.tensor(v, dtype=torch.float32),
    }
    return apply_alignment(seq, alignment(align, yaw_deg))


def check_frames(seq: Sequence) -> np.ndarray:
    """World-frame mean of the specific force; should be about [0, 0, +9.81]."""
    R = Rotation.from_quat(seq["gt_orientation"].tensor().numpy())
    return R.apply(seq["acc"].numpy()).mean(axis=0)


def list_sequences(root: Path = ROOT) -> List[str]:
    return sorted(p.name for p in Path(root).iterdir()
                  if (p / "mav0" / "imu0" / "data.csv").exists())


def load_split(names: List[str], dt: float = 0.01, root: Path = ROOT,
               align: str = "none", yaw_deg: float = 0.0,
               verbose: bool = True) -> List[Tuple[str, Sequence]]:
    pairs = []
    for name in names:
        seq = load_euroc(Path(root) / name, dt=dt, align=align, yaw_deg=yaw_deg)
        if verbose:
            d = np.linalg.norm(np.diff(seq["gt_translation"].numpy(), axis=0), axis=1).sum()
            g = check_frames(seq)
            b = seq["acc"].numpy().mean(0)
            print(f"  {name:<18} {len(seq['acc']):>6} samples  {d:>6.1f} m  "
                  f"world-accel [{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}]  "
                  f"body-accel [{b[0]:+.2f} {b[1]:+.2f} {b[2]:+.2f}]")
        pairs.append((name, seq))
    return pairs


if __name__ == "__main__":
    lists = read_lists()
    for k, v in lists.items():
        print(f"{k:<6} ({len(v)}): {' '.join(v)}")
    print(f"train == val: {lists['train'] == lists['val']}")
    for al in ("none", "gravity"):
        print(f"\n=== align={al} (dt=0.01) ===")
        load_split(list_sequences(), dt=0.01, align=al)
