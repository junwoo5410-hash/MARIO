#!/usr/bin/env python
"""Stage 3 — coordinate-frame verification.

MARIO_closedloop_prompt.md 함정 D says the Blackbird constants must not be reused for
PX4 and that a PX4 transform has to be derived with its reasoning written down. This
script does both, using no model at all (spec: "모델을 쓰지 않는다").

--------------------------------------------------------------------------------------
What MARIO's frames actually are — measured, not assumed
--------------------------------------------------------------------------------------
Both facts below are re-measured by test 2 on every run rather than trusted:

WORLD (W_M):  z is UP.  Rotating the measured specific force into the world frame over a
              whole flight averages to (0, 0, +9.75) = -g, so g points along -z. The GT
              altitude channel is positive (1.17 .. 1.79 m indoors). Combined with
              data.py's R_W_NED = diag(1, -1, -1) applied to a NED source, W_M is NWU
              (x North, y West, z Up).

BODY  (B_M):  x = Right, y = Back, z = Down.  In the yawForward flights the vehicle noses
              along its velocity, and the body-frame velocity is dominated by -y
              (e.g. egg: (0.12, -0.93, -0.15) as a unit vector). So forward = -y. With
              z = Down (mean acc_z = -11.2, i.e. gravity's reaction along -z) a
              right-handed frame forces x = y x z = Right.
              This reproduces data.py's R_B_I exactly: R_B_I = C_B^T (test 3 asserts it).

--------------------------------------------------------------------------------------
PX4 -> MARIO transform (함정 D)
--------------------------------------------------------------------------------------
PX4 uses a NED world and an FRD body (x Forward, y Right, z Down), and
vehicle_attitude.q is the rotation FRD -> NED.

    C_W : NED -> NWU      = diag(1, -1, -1)
          (north kept, east -> west, down -> up; an improper-looking but proper rotation:
           it is a 180 deg turn about the north axis, det = +1)

    C_B : FRD -> B_M      = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
          columns are the images of the FRD basis:
            Forward (1,0,0) -> (0,-1,0) = -y_M   (forward is -y, as measured)
            Right   (0,1,0) -> (1,0,0)  = +x_M
            Down    (0,0,1) -> (0,0,1)  = +z_M
          i.e. a -90 deg rotation about the shared down axis, det = +1.

Feeding the network:
    acc_M   = C_B @ acc_FRD
    gyro_M  = C_B @ gyro_FRD
    R_M     = C_W @ R_ned_frd @ C_B.T      # B_M -> NWU, then .Log() for rot_so3

Reading the output back (disp_M is in B_M):
    disp_NED = R_ned_frd @ C_B.T @ disp_M
    disp_FRD = C_B.T @ disp_M              # body-frame form, if publishing body velocity

Note the world hop cancels: C_W.T @ (C_W @ R_ned_frd @ C_B.T) == R_ned_frd @ C_B.T.
The world convention therefore only matters for the attitude the network is fed, never
for converting its output. Test 4 checks this on real flight data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mario.data import R_B_I, R_W_NED, load_blackbird  # noqa: E402

WINDOW_STRIDE = 9  # LABEL_STRIDE
ROUNDTRIP_TOL = 1e-4  # spec: 복원 궤적 ATE < 1e-4 m

# ---- the two constants derived in the docstring -------------------------------------
C_W = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # NED -> NWU
C_B = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # FRD -> B_M

EVAL_ROOT = Path("/src/gs25122/blackbird_data/eval")
OK, FAIL = "  [ok]  ", "  [FAIL]"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def is_rotation(m: np.ndarray) -> bool:
    return np.allclose(m @ m.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(m), 1.0)


# --------------------------------------------------------------------------------------
def test_roundtrip(sequences: dict) -> dict:
    """Spec test: world -> body -> world must be lossless, so any later error is the model."""
    section("Test 1 — GT 왕복 변환 (world -> body -> world)")
    print(f"{'trajectory':<14}{'poses':>8}{'ATE [m]':>14}{'max err [m]':>14}")
    print("-" * 50)

    out = {}
    for name, data in sequences.items():
        pos, rot = data["gt_translation"], data["gt_orientation"]
        idx = list(range(14, len(pos) - WINDOW_STRIDE, WINDOW_STRIDE))

        recovered = [pos[idx[0]].clone()]
        for t in idx:
            world_disp = pos[t + WINDOW_STRIDE] - pos[t]
            body_disp = rot[t].Inv() @ world_disp  # dataset.py:31
            recovered.append(recovered[-1] + (rot[t] @ body_disp))  # evaluate.py:50

        rec = torch.stack(recovered).numpy()
        gt = pos[[idx[0]] + [t + WINDOW_STRIDE for t in idx]].numpy()
        err = np.linalg.norm(rec - gt, axis=1)
        ate = float(np.sqrt(np.mean(err**2)))
        out[name] = {"poses": len(rec), "ate": ate, "max_err": float(err.max())}
        print(f"{name:<14}{len(rec):>8}{ate:>14.3e}{err.max():>14.3e}")

    worst = max(v["ate"] for v in out.values())
    print("-" * 50)
    status = OK if worst < ROUNDTRIP_TOL else FAIL
    print(f"{status} worst ATE {worst:.3e} (기준 < {ROUNDTRIP_TOL:.0e})")
    return {"per_trajectory": out, "worst_ate": worst, "passed": worst < ROUNDTRIP_TOL}


# --------------------------------------------------------------------------------------
def test_conventions(sequences: dict) -> dict:
    """Re-derive MARIO's world/body conventions from the data instead of trusting comments."""
    section("Test 2 — MARIO 좌표 규약 실측")

    grav, fwd = [], []
    for data in sequences.values():
        R = data["gt_orientation"].matrix().numpy()
        grav.append(np.einsum("tij,tj->ti", R, data["acc"].numpy()).mean(0))

        v = data["velocity"].numpy()
        moving = np.linalg.norm(v, axis=1) > 1.0
        vb = np.einsum("tji,tj->ti", R[moving], v[moving])  # world -> body
        fwd.append((vb / np.linalg.norm(vb, axis=1, keepdims=True)).mean(0))

    g_mean, f_mean = np.mean(grav, axis=0), np.mean(fwd, axis=0)
    world_up = g_mean[2] > 0
    body_down = np.mean([d["acc"][:, 2].mean().item() for d in sequences.values()]) < 0
    fwd_axis = int(np.argmax(np.abs(f_mean)))
    fwd_is_neg_y = fwd_axis == 1 and f_mean[1] < 0

    print(f"  mean(R @ acc_body) = {np.round(g_mean, 3)}  ->  -g_world")
    print(f"{OK if world_up else FAIL} world z = UP (NWU)   [g_z = {-g_mean[2]:+.3f}]")
    print(f"{OK if body_down else FAIL} body  z = DOWN (FRD류)")
    print(f"  yawForward 바디 속도 단위벡터 평균 = {np.round(f_mean, 3)}")
    print(f"{OK if fwd_is_neg_y else FAIL} body forward = -y  ->  body = (x Right, y Back, z Down)")

    passed = bool(world_up and body_down and fwd_is_neg_y)
    return {
        "mean_neg_gravity_world": g_mean.tolist(),
        "mean_body_velocity_dir": f_mean.tolist(),
        "world_z_up": bool(world_up),
        "body_z_down": bool(body_down),
        "forward_is_minus_y": bool(fwd_is_neg_y),
        "passed": passed,
    }


# --------------------------------------------------------------------------------------
def test_px4_constants() -> dict:
    """C_W / C_B must be proper rotations and must agree with the Blackbird constants."""
    section("Test 3 — PX4 변환 상수 (함정 D)")

    checks = {
        "C_W is a proper rotation": is_rotation(C_W),
        "C_B is a proper rotation": is_rotation(C_B),
        # data.py's R_B_I maps B_M -> raw FRD body, i.e. exactly the inverse of C_B.
        "C_B.T == R_B_I (data.py)": np.allclose(C_B.T, R_B_I),
        # data.py's R_W_NED maps the NED source world into NWU — the same hop as C_W.
        "C_W == R_W_NED (data.py)": np.allclose(C_W, R_W_NED),
        # Forward in FRD must land on -y in B_M.
        "C_B @ Forward == -y_M": np.allclose(C_B @ [1, 0, 0], [0, -1, 0]),
        "C_B @ Right   == +x_M": np.allclose(C_B @ [0, 1, 0], [1, 0, 0]),
        "C_B @ Down    == +z_M": np.allclose(C_B @ [0, 0, 1], [0, 0, 1]),
        # Gravity sanity: NED gravity (0,0,+9.81) must become (0,0,-9.81) in NWU.
        "C_W @ g_NED == g_NWU": np.allclose(C_W @ [0, 0, 9.81], [0, 0, -9.81]),
    }
    for label, ok in checks.items():
        print(f"{OK if ok else FAIL} {label}")
    return {"checks": {k: bool(v) for k, v in checks.items()}, "passed": all(checks.values()),
            "C_W": C_W.tolist(), "C_B": C_B.tolist()}


# --------------------------------------------------------------------------------------
def test_px4_pipeline(sequences: dict) -> dict:
    """End-to-end: re-express real flights in PX4 convention, run the adapter, compare.

    Blackbird is re-expressed as if PX4 had produced it (NED world, FRD body, q = FRD->NED),
    then pushed through the transforms the ROS node will use. The recovered NED displacement
    must equal the original world displacement mapped into NED.
    """
    section("Test 4 — PX4 -> MARIO -> NED 파이프라인 (실제 비행 데이터)")
    print(f"{'trajectory':<14}{'acc [m/s2]':>14}{'attitude':>12}{'disp [m]':>14}")
    print("-" * 54)

    out = {}
    for name, data in sequences.items():
        R_M = data["gt_orientation"].matrix().numpy()  # B_M -> NWU
        acc_M = data["acc"].numpy()
        pos_NWU = data["gt_translation"].numpy()

        # ---- pretend PX4 produced this flight -------------------------------------
        acc_FRD = np.einsum("ij,tj->ti", C_B.T, acc_M)  # B_M -> FRD
        R_ned_frd = np.einsum("ij,tjk,kl->til", C_W.T, R_M, C_B)  # FRD -> NED
        pos_NED = np.einsum("ij,tj->ti", C_W.T, pos_NWU)

        # ---- the adapter the node will run ----------------------------------------
        acc_back = np.einsum("ij,tj->ti", C_B, acc_FRD)  # feed the net
        R_back = np.einsum("ij,tjk,kl->til", C_W, R_ned_frd, C_B.T)  # feed the net

        # a stand-in for the network output: the true body-frame displacement
        t = np.arange(14, len(pos_NWU) - WINDOW_STRIDE, WINDOW_STRIDE)
        world_disp = pos_NWU[t + WINDOW_STRIDE] - pos_NWU[t]
        disp_M = np.einsum("tji,tj->ti", R_M[t], world_disp)  # NWU -> B_M

        # output conversion used by the node
        disp_NED = np.einsum("tij,jk,tk->ti", R_ned_frd[t], C_B.T, disp_M)
        expected_NED = np.einsum("ij,tj->ti", C_W.T, world_disp)

        e_acc = float(np.abs(acc_back - acc_M).max())
        e_att = float(np.abs(R_back - R_M).max())
        e_disp = float(np.linalg.norm(disp_NED - expected_NED, axis=1).max())
        out[name] = {"acc_err": e_acc, "attitude_err": e_att, "disp_err": e_disp}
        print(f"{name:<14}{e_acc:>14.3e}{e_att:>12.3e}{e_disp:>14.3e}")

    worst = max(max(v.values()) for v in out.values())
    print("-" * 54)
    status = OK if worst < 1e-5 else FAIL
    print(f"{status} 최대 오차 {worst:.3e} (수치 정밀도 수준이어야 함)")
    return {"per_trajectory": out, "worst_err": worst, "passed": worst < 1e-5}


# --------------------------------------------------------------------------------------
def main() -> int:
    # glob can pick up __MACOSX / .DS_Store junk shipped in the archive
    paths = sorted(d for d in EVAL_ROOT.glob("*/yawForward/*") if (d / "imu_data.csv").exists())
    if not paths:
        print(f"{FAIL} no sequences under {EVAL_ROOT}")
        return 1
    print(f"loading {len(paths)} eval sequences ...")
    sequences = {p.parts[-3]: load_blackbird(p) for p in paths}

    report = {
        "roundtrip": test_roundtrip(sequences),
        "conventions": test_conventions(sequences),
        "px4_constants": test_px4_constants(),
        "px4_pipeline": test_px4_pipeline(sequences),
    }

    section("Stage 3 결과")
    all_passed = all(r["passed"] for r in report.values())
    for key, res in report.items():
        print(f"{OK if res['passed'] else FAIL} {key}")
    report["passed"] = all_passed

    out = REPO_ROOT / "mario_sitl" / "results" / "frames_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  report: {out}")
    print(f"\n{OK if all_passed else FAIL} Stage 3 {'통과 — Stage 4 진행 가능' if all_passed else '실패 — 여기서 해결할 것'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
