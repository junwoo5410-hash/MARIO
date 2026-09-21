#!/usr/bin/env python
"""Loader for the UZH-FPV Drone Racing dataset, shaped like euroc_data.py.

Why this dataset. EuRoC turned out to be outside what this estimator can do at all --
0.4-0.5 m/s, gyro an order of magnitude quieter than Blackbird, no acceleration events --
so every arm there sat below the stationary baseline and the transfer study could only
rank methods, never show one working. UZH-FPV is real flight by an FPV racing pilot:
indoor_forward_3 measures 5.80 m/s mean (9.29 m/s peak), |accel| 6.39 m/s^2, 100 % of
moving samples above 1 m/s^2, drag signal 2.42 -- at or above Blackbird's regime on every
channel. It is the dataset where "a few minutes of target data is enough" can be tested.

Source (CC BY-NC-SA 3.0, no registration):
http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/<seq>_snapdragon_with_gt.zip
Only the 16 sequences with public ground truth are usable; the nine benchmark sequences
and the SplitS track keep their ground truth withheld.

File layout inside each zip (the site documents the rosbag variant only; these columns
were read off the files):

    imu.txt          # id timestamp ang_vel_{x,y,z} lin_acc_{x,y,z}      500 Hz
    groundtruth.txt  # timestamp t{x,y,z} q{x,y,z,w}                     500 Hz

Two things the EuRoC loader did not have to handle:

  * ground truth covers less than the IMU recording (indoor_forward_3: 49.5 s of Leica
    inside 92.1 s of flight), so the usable span is the overlap, not the file;
  * the Leica loses the prism. A gap interpolated across would be invented motion, so
    ``load_uzhfpv`` keeps the longest gap-free span and reports what it dropped.

Mounting differs from Blackbird as it did for EuRoC -- the measured body-frame gravity
direction is roughly z-flipped -- so the same gravity+yaw alignment applies. The
direction is measured on TRAINING sequences only and cached to results/uzhfpv/
gravity_dir.json by ``python uzhfpv_data.py --measure``; nothing here hardcodes a
number that was not measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pypose as pp
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path("/src/gs25122/uzhfpv/seq")
RESULTS = Path(__file__).resolve().parents[1] / "results" / "uzhfpv"
GRAVITY_FILE = RESULTS / "gravity_dir.json"

#: same constant euroc_data.py aligns onto, measured on Blackbird's five SEEN trajectories
BLACKBIRD_GRAVITY_DIR = np.array([-0.0097, 0.1089, -0.9940])
#: a ground-truth gap longer than this is a tracker dropout, not a sample spacing
MAX_GT_GAP_S = 0.1

Sequence = Dict[str, torch.Tensor]

#: seconds of ground truth dropped as tracker gaps, per sequence directory name.
#: Kept beside the sequences rather than inside them: every consumer of a Sequence
#: slices all of its entries along time (finetune_transfer.split_tail does), so a
#: scalar stored in there raises "slice() cannot be applied to a 0-dim tensor".
DROPPED_SECONDS: Dict[str, float] = {}


def _shortest_arc(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3) if u @ v > 0 else -np.eye(3)
    return Rotation.from_rotvec(axis / s * np.arctan2(s, u @ v)).as_matrix()


def gravity_dir() -> np.ndarray:
    """The measured UZH-FPV body-frame gravity direction (see ``--measure``)."""
    if not GRAVITY_FILE.exists():
        raise FileNotFoundError(
            f"{GRAVITY_FILE} missing -- run `python uzhfpv_data.py --measure` first; "
            "it measures the direction on the training split only")
    return np.array(json.loads(GRAVITY_FILE.read_text())["gravity_dir"])


def alignment(mode: str, yaw_deg: float = 0.0) -> np.ndarray:
    """``none`` keeps the dataset frame, ``yaw`` spins it about its own z (no flip),
    ``gravity`` maps the measured gravity direction onto Blackbird's and then spins.

    The third family is the one euroc_data uses. The second exists because on UZH-FPV
    the un-flipped frame scores better zero-shot than any gravity-aligned yaw, so the
    rotation that helps may be a plain yaw rather than the 170-degree flip.
    """
    if mode == "none":
        return np.eye(3)
    if mode == "yaw":
        return Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    if mode != "gravity":
        raise ValueError(f"unknown alignment {mode!r}")
    C = _shortest_arc(gravity_dir(), BLACKBIRD_GRAVITY_DIR)
    if yaw_deg:
        C = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ C
    return C


def apply_alignment(seq: Sequence, C: np.ndarray) -> Sequence:
    if np.allclose(C, np.eye(3)):
        return seq
    Ct = torch.tensor(C.T, dtype=torch.float32)
    seq["acc"] = seq["acc"] @ Ct
    seq["gyro"] = seq["gyro"] @ Ct
    R = Rotation.from_quat(seq["gt_orientation"].tensor().numpy()).as_matrix() @ C.T
    seq["gt_orientation"] = pp.SO3(
        torch.tensor(Rotation.from_matrix(R).as_quat(), dtype=torch.float32))
    return seq


def _longest_span(t_gt: np.ndarray) -> Tuple[float, float, float]:
    """Longest run of ground truth with no gap over ``MAX_GT_GAP_S``; also the time dropped."""
    breaks = np.where(np.diff(t_gt) > MAX_GT_GAP_S)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(t_gt) - 1]
    spans = t_gt[ends] - t_gt[starts]
    i = int(np.argmax(spans))
    return float(t_gt[starts[i]]), float(t_gt[ends[i]]), float(spans.sum() - spans[i])


def load_uzhfpv(seq_dir: Path, dt: float = 0.01, align: str = "none",
                yaw_deg: float = 0.0) -> Sequence:
    """Load one sequence onto a uniform ``dt`` grid, restricted to gap-free ground truth."""
    seq_dir = Path(seq_dir)
    imu = np.loadtxt(seq_dir / "imu.txt")
    gt = np.loadtxt(seq_dir / "groundtruth.txt")

    t_imu, gyro_raw, acc_raw = imu[:, 1], imu[:, 2:5], imu[:, 5:8]
    t_gt, pos, quat_xyzw = gt[:, 0], gt[:, 1:4], gt[:, 4:8]

    g0, g1, dropped = _longest_span(t_gt)
    t0 = max(t_imu[0], g0)
    t1 = min(t_imu[-1], g1)
    times = np.arange(t0, t1 - dt - 1e-9, dt)
    if len(times) < 1000:
        raise ValueError(f"{seq_dir.name}: only {len(times)} samples of usable ground truth")

    gyro = interp1d(t_imu, gyro_raw, axis=0)(times)
    acc = interp1d(t_imu, acc_raw, axis=0)(times)
    p = interp1d(t_gt, pos, axis=0)(times)
    q = Slerp(t_gt, Rotation.from_quat(quat_xyzw))(times).as_quat()
    # UZH-FPV ships no velocity channel; differentiate the Leica position instead
    v = np.gradient(p, dt, axis=0)

    seq = {
        "time": torch.tensor(times, dtype=torch.float64),
        "acc": torch.tensor(acc, dtype=torch.float32),
        "gyro": torch.tensor(gyro, dtype=torch.float32),
        "gt_translation": torch.tensor(p, dtype=torch.float32),
        "gt_orientation": pp.SO3(torch.tensor(q, dtype=torch.float32)),
        "velocity": torch.tensor(v, dtype=torch.float32),
    }
    DROPPED_SECONDS[seq_dir.name] = dropped
    return apply_alignment(seq, alignment(align, yaw_deg))


def check_frames(seq: Sequence) -> np.ndarray:
    R = Rotation.from_quat(seq["gt_orientation"].tensor().numpy())
    return R.apply(seq["acc"].numpy()).mean(axis=0)


def list_sequences(root: Path = ROOT) -> List[str]:
    return sorted(p.name for p in Path(root).iterdir() if (p / "imu.txt").exists())


def read_lists(root: Path = ROOT) -> Dict[str, List[str]]:
    """Train/test split. UZH-FPV publishes none for this purpose, so it is ours."""
    out = {}
    for name in ("train", "test"):
        path = Path(root).parent / f"{name}_list.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- write the split before training")
        out[name] = [ln.strip() for ln in path.read_text().split() if ln.strip()]
    return out


def load_split(names: List[str], dt: float = 0.01, root: Path = ROOT,
               align: str = "none", yaw_deg: float = 0.0,
               verbose: bool = True) -> List[Tuple[str, Sequence]]:
    pairs = []
    for name in names:
        seq = load_uzhfpv(Path(root) / name, dt=dt, align=align, yaw_deg=yaw_deg)
        if verbose:
            xyz = seq["gt_translation"].numpy()
            d = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
            sp = np.linalg.norm(seq["velocity"].numpy(), axis=1)
            g, b = check_frames(seq), seq["acc"].numpy().mean(0)
            print(f"  {name:<20} {len(seq['acc']):>6} samples  {len(seq['acc']) * dt:>6.1f} s  "
                  f"{d:>6.1f} m  mean {sp.mean():.2f} m/s  max {sp.max():.2f}  "
                  f"world-accel [{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}]  "
                  f"body-accel [{b[0]:+.2f} {b[1]:+.2f} {b[2]:+.2f}]  "
                  f"dropped {DROPPED_SECONDS.get(name, 0.0):.1f} s")
        pairs.append((name, seq))
    return pairs


def measure_gravity_dir(names: List[str], root: Path = ROOT) -> dict:
    """Mean body-frame specific-force direction over ``names``, plus its scatter.

    Only training sequences may be passed: this constant feeds the alignment that every
    later number depends on, and measuring it on a test sequence would leak.
    """
    dirs = {}
    for name in names:
        d = load_uzhfpv(Path(root) / name)["acc"].numpy().mean(0)
        dirs[name] = (d / np.linalg.norm(d)).tolist()
    mean = np.mean(list(dirs.values()), axis=0)
    mean /= np.linalg.norm(mean)
    spread = [float(np.degrees(np.arccos(np.clip(np.array(d) @ mean, -1, 1))))
              for d in dirs.values()]
    return {"gravity_dir": mean.tolist(), "per_sequence": dirs,
            "max_scatter_deg": max(spread), "sequences": names,
            "angle_to_blackbird_deg": float(np.degrees(np.arccos(
                np.clip(mean @ BLACKBIRD_GRAVITY_DIR /
                        np.linalg.norm(BLACKBIRD_GRAVITY_DIR), -1, 1))))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="measure the body gravity direction on the training split and cache it")
    ap.add_argument("--align", default="none", choices=("none", "yaw", "gravity"))
    ap.add_argument("--dt", type=float, default=0.01)
    args = ap.parse_args()

    if args.measure:
        names = read_lists()["train"]
        info = measure_gravity_dir(names)
        RESULTS.mkdir(parents=True, exist_ok=True)
        GRAVITY_FILE.write_text(json.dumps(info, indent=2))
        print(f"train sequences: {' '.join(names)}")
        print(f"gravity dir {np.round(info['gravity_dir'], 5)}  "
              f"scatter {info['max_scatter_deg']:.2f} deg  "
              f"vs Blackbird {info['angle_to_blackbird_deg']:.1f} deg")
        print(f"wrote {GRAVITY_FILE}")
        return 0

    print(f"=== align={args.align} dt={args.dt} ===")
    load_split(list_sequences(), dt=args.dt, align=args.align)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
