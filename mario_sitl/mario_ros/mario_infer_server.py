#!/usr/bin/env python
"""MARIO inference server, Unix-domain socket, runs in the ``mamba`` conda env.

Why a separate process: the ROS 2 stack is a py3.11 RoboStack env, mamba-ssm is a cp310
wheel, and this host's GLIBC 2.31 is below the 2.32 that prebuilt cp311 wheels need, so
rclpy and mamba_ssm cannot share one interpreter. The socket hop is local and measured as
part of the end-to-end latency the node reports, so it is not hidden cost.

Protocol, little-endian, one request per connection-less exchange over SOCK_SEQPACKET:
    request : 4s magic 'MRQ0' | uint32 n_imu | uint32 n_att
              | float64[n_imu] imu_t | float32[n_imu*3] gyro | float32[n_imu*3] acc
              | float64[n_att] att_t | float32[n_att*4] q_wxyz
    reply   : 4s magic 'MRP0' | uint8 ok | float32[3] vel_ned | float32[3] disp_body
              | float32[3] cov | float64 t_ref | float64 infer_ms
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mario_frames as mf  # noqa: E402
from mario.model import CausalMambaDispNet  # noqa: E402

REQ_MAGIC = b"MRQ0"
REP_MAGIC = b"MRP0"
MAX_MSG = 1 << 20


def load_net(ckpt: Path, device: torch.device) -> CausalMambaDispNet:
    net = CausalMambaDispNet().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()  # load-bearing: BatchNorm in train() mode makes out[0] depend on the
                # whole window, which would break causality. eval() is causal.
    return net


def unpack_request(buf: memoryview) -> dict:
    if bytes(buf[:4]) != REQ_MAGIC:
        raise ValueError("bad magic")
    n_imu, n_att = struct.unpack_from("<II", buf, 4)
    o = 12
    imu_t = np.frombuffer(buf, np.float64, n_imu, o); o += n_imu * 8
    gyro = np.frombuffer(buf, np.float32, n_imu * 3, o).reshape(n_imu, 3); o += n_imu * 12
    acc = np.frombuffer(buf, np.float32, n_imu * 3, o).reshape(n_imu, 3); o += n_imu * 12
    att_t = np.frombuffer(buf, np.float64, n_att, o); o += n_att * 8
    quat = np.frombuffer(buf, np.float32, n_att * 4, o).reshape(n_att, 4)
    return {"imu_t": imu_t, "gyro": gyro, "acc": acc, "att_t": att_t, "quat": quat}


def infer(net, req: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    imu_t, gyro_in, acc_in = mf.make_monotonic(req["imu_t"], req["gyro"], req["acc"])
    att_t, quat_in = mf.make_monotonic(req["att_t"], req["quat"])
    # Newest instant both sensors actually cover; never extrapolate the window forward.
    t_end = min(imu_t[-1], att_t[-1])
    grid = t_end - (mf.WINDOW - 1 - np.arange(mf.WINDOW)) * mf.DT
    if grid[0] < imu_t[0]:
        raise ValueError(f"buffer too short: need {mf.WINDOW * mf.DT:.2f}s, "
                         f"have {imu_t[-1] - imu_t[0]:.2f}s")

    gyro, acc, R_ned_frd = mf.resample(imu_t, gyro_in, acc_in, att_t, quat_in, grid)
    x = mf.build_inputs(gyro, acc, R_ned_frd, device)

    with torch.no_grad():
        disp, cov = net(x["acc"], x["gyro"], x["rot_so3"])
    i = mf.OUTPUT_INDEX
    disp_body = disp[0, i].float().cpu().numpy().astype(np.float64)
    cov_out = cov[0, i].float().cpu().numpy().astype(np.float64)

    # 결정 A: report a velocity, do not integrate here. The step spans LABEL_STRIDE
    # samples, so dt = 9 * 0.01 = 0.09 s.
    step_dt = mf.LABEL_STRIDE * mf.DT
    idx = mf.LABEL_START + i * mf.LABEL_STRIDE          # start index of this step
    vel_ned = mf.disp_to_ned(disp_body, R_ned_frd[min(idx, mf.WINDOW - 1)]) / step_dt
    t_ref = grid[min(idx, mf.WINDOW - 1)]
    return vel_ned, disp_body, cov_out, t_ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/mario_infer.sock")
    # Same checkpoint the Stage 0 scripts use. Note that
    # ~/causal_mamba_disp_trial8_results/best.pt is a leftover from a stubbed-Mamba smoke
    # test and will not load into the real network.
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).resolve().parents[2] / "runs" / "nm_s42" / "best.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_net(args.ckpt, device)

    # Warm up: the first CUDA launch pays kernel compilation and allocator setup, which
    # would otherwise land in the p95 the node reports.
    dummy = {k: torch.zeros(1, mf.WINDOW, 3, device=device) for k in ("a", "g", "r")}
    for _ in range(5):
        with torch.no_grad():
            net(dummy["a"], dummy["g"], dummy["r"])
    torch.cuda.synchronize() if device.type == "cuda" else None

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    srv.bind(args.socket)
    srv.listen(4)
    print(f"ready ckpt={args.ckpt} device={device} socket={args.socket}", flush=True)

    # Serve clients one after another: a finished node closing its socket must not take
    # the server (and its warmed-up CUDA context) down with it.
    while True:
        conn, _ = srv.accept()
        print("client connected", flush=True)
        serve(conn, net, device)
        conn.close()
        print("client disconnected", flush=True)
    return 0


def serve(conn: socket.socket, net, device: torch.device) -> None:
    while True:
        data = conn.recv(MAX_MSG)
        if not data:
            return
        t0 = time.perf_counter()
        try:
            req = unpack_request(memoryview(data))
            vel, disp, cov, t_ref = infer(net, req, device)
            ok = 1
        except Exception as exc:  # report, never crash the flight-side node
            print(f"infer failed: {exc}", flush=True)
            vel = disp = cov = np.zeros(3)
            t_ref, ok = 0.0, 0
        ms = (time.perf_counter() - t0) * 1e3
        conn.send(REP_MAGIC + struct.pack("<B", ok)
                  + vel.astype(np.float32).tobytes()
                  + disp.astype(np.float32).tobytes()
                  + cov.astype(np.float32).tobytes()
                  + struct.pack("<dd", t_ref, ms))


if __name__ == "__main__":
    raise SystemExit(main())
