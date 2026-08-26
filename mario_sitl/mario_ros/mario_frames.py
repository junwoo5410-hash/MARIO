"""PX4 <-> MARIO frame adapter and window builder.

Runs in the ``mamba`` conda env (torch + pypose + scipy). Everything MARIO-specific
lives here so the ROS node stays a thin, numpy-only collector.

Frames, all four verified in Stage 3 (``scripts/verify_frames.py``, max round-trip
error 2.4e-07 m):

    PX4    world = NED,  body = FRD,  q = Hamilton (w,x,y,z), FRD -> NED
    MARIO  world = NWU,  body = (x Right, y Back, z Down)

so ``C_W`` maps NED -> NWU and ``C_B`` maps FRD -> B_M.
"""

from __future__ import annotations

import numpy as np
import pypose as pp
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

C_W = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # NED -> NWU
C_B = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # FRD -> B_M

DT = 0.01           # training grid: 100 Hz (mario.data.load_blackbird)
WINDOW = 1000       # samples per inference window
LABEL_STRIDE = 9    # encoder downsampling: two stride-3 convs
LABEL_START = 14

# The network emits 112 steps for a 1000-sample window, but out[i]'s receptive field
# ends at input 9i+12, so out[110] and out[111] are clipped by zero padding at the
# window edge (measured: both report a field ending exactly at 999 instead of their
# natural 1002/1011). out[109] is the freshest padding-free output; it predicts the
# displacement over input samples 995..1004.
OUTPUT_INDEX = 109

# Blackbird's motor ch2 is body-z mass-normalised collective thrust, |.| <= 13 m/s^2,
# and corr(ch2, acc_z) = +0.92. Feeding zeros instead costs +921% ATE (spec 함정 A).
ACCZ_SCALE = 13.0


def px4_quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """(N,4) Hamilton (w,x,y,z) -> (N,3,3) rotation FRD -> NED."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


#: Samples further than this from the median timestamp are from a different clock base.
#: Recordings are minutes long, so an hour is a wide margin that still catches the
#: pre-sync stamps without assuming any particular epoch.
MAX_CLOCK_SPAN_S = 3600.0


def make_monotonic(t: np.ndarray, *arrays: np.ndarray):
    """Give the caller a usable time base: one clock, strictly increasing.

    Two distinct problems, both observed on this PX4 build:

    * Duplicate or out-of-order timestamps (best-effort QoS, and the FMU republishes on
      reset). ``Slerp`` and ``interp1d`` reject those outright -- in the first Stage 2 run
      this killed 297 of ~2000 inferences with "Times must be in strictly increasing
      order".
    * A handful of samples per stream stamped on a different clock. PX4 stamps with raw
      hrt time until ``uxrce_dds_client`` applies the agent's time offset, so the first
      few messages of a recording carry boot-relative stamps (~1e3 s) while the rest carry
      epoch-offset ones (~1.8e9 s). Four to ten samples per stream, enough that one flight
      asked for a 1.79e9 s resampling grid and tried to allocate 1.3 TiB.
    """
    t = np.asarray(t, dtype=np.float64)
    if len(t) > 2:
        keep_clock = np.abs(t - np.median(t)) <= MAX_CLOCK_SPAN_S
        if not keep_clock.all():
            t = t[keep_clock]
            arrays = tuple(a[keep_clock] for a in arrays)

    order = np.argsort(t, kind="stable")
    t = t[order]
    arrays = tuple(a[order] for a in arrays)
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 0
    return (t[keep],) + tuple(a[keep] for a in arrays)


def resample(
    imu_t: np.ndarray, gyro: np.ndarray, acc: np.ndarray,
    att_t: np.ndarray, q_wxyz: np.ndarray, grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put IMU and attitude on a common uniform grid.

    Mirrors ``mario.data.load_blackbird``: linear ``interp1d`` for the IMU channels and
    ``Slerp`` for attitude. Linear interpolation of raw quaternions is not equivalent and
    is explicitly forbidden by the spec.
    """
    gyro_g = interp1d(imu_t, gyro, axis=0, bounds_error=False, fill_value="extrapolate")(grid)
    acc_g = interp1d(imu_t, acc, axis=0, bounds_error=False, fill_value="extrapolate")(grid)

    rot = Rotation.from_quat(np.asarray(q_wxyz, dtype=np.float64)[:, [1, 2, 3, 0]])
    # Slerp cannot extrapolate, so clamp the grid into the attitude's own time span.
    clamped = np.clip(grid, att_t[0], att_t[-1])
    R_ned_frd = Slerp(att_t, rot)(clamped).as_matrix()
    return gyro_g, acc_g, R_ned_frd


def build_inputs(gyro_frd: np.ndarray, acc_frd: np.ndarray, R_ned_frd: np.ndarray,
                 device: torch.device) -> dict:
    """PX4-frame window -> the four (1, T, 3) tensors the network takes."""
    acc_m = np.einsum("ij,tj->ti", C_B, acc_frd)
    gyro_m = np.einsum("ij,tj->ti", C_B, gyro_frd)
    R_m = np.einsum("ij,tjk,kl->til", C_W, R_ned_frd, C_B.T)  # B_M -> NWU

    quat_xyzw = Rotation.from_matrix(R_m).as_quat()
    rot_so3 = pp.SO3(torch.tensor(quat_xyzw, dtype=torch.float32)).Log().tensor()

    motor = np.zeros_like(acc_m)
    motor[:, 2] = np.clip(acc_m[:, 2] / ACCZ_SCALE, -1.0, 1.0)

    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32).unsqueeze(0).to(device)

    return {"acc": t(acc_m), "gyro": t(gyro_m), "motor": t(motor),
            "rot_so3": rot_so3.unsqueeze(0).to(device), "R_ned_frd": R_ned_frd}


def disp_to_ned(disp_m: np.ndarray, R_ned_frd_at: np.ndarray) -> np.ndarray:
    """Body-frame displacement -> NED, using the attitude at the step's own start index."""
    return R_ned_frd_at @ C_B.T @ np.asarray(disp_m, dtype=np.float64)
