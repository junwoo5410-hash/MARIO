#!/usr/bin/env python
"""Stage 2 — MARIO real-time shadow node. Logs only; publishes nothing.

The point of shadow mode is to answer one question before any closed-loop attempt:
can MARIO produce estimates fast enough to fly on? Success criteria from the spec are
>=20 Hz sustained inference and p95 end-to-end latency <=50 ms.

"End to end" is measured honestly: from the arrival timestamp of the newest IMU sample
that entered the window, to the moment this node holds the resulting velocity. That
includes the socket hop to the inference server, resampling, and the network itself.

Nothing here touches control. Ground truth is recorded for scoring only
(spec §1: "Gazebo GT: 검증용으로만 기록, 제어에 사용 금지").
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import (SensorCombined, VehicleAttitude, VehicleLocalPosition,
                          VehicleOdometry)
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)

STEP_DT = 0.09          # LABEL_STRIDE * DT: the interval one displacement step spans
WINDOW_SECONDS = 10.0   # 1000 samples @ 100 Hz
BUFFER_SECONDS = 12.0   # a little slack so resampling never runs off the end
REQ_MAGIC = b"MRQ0"
REP_MAGIC = b"MRP0"


class MarioOdometryNode(Node):
    def __init__(self, sock_path: str, rate_hz: float, duration: float, out_path: Path,
                 publish: bool = False):
        super().__init__("mario_odometry_node")
        self.out_path = out_path
        self.publish_ev = publish
        self.duration = duration
        self.rate_hz = rate_hz

        self.imu: deque = deque()
        self.att: deque = deque()
        self.gt = [math.nan] * 3
        self.gt_vel = [math.nan] * 3
        self.est_vel = [math.nan] * 3
        self.last_imu_wall = None

        self.finished = False
        self.records: list = []
        self.skipped = 0
        self.failed = 0
        self.t_start = time.perf_counter()

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock.connect(sock_path)
        self.get_logger().info(f"connected to inference server at {sock_path}")

        # Stage 5 turns this on; Stage 2 leaves it off and the node stays a pure observer.
        self.pub_ev = self.create_publisher(
            VehicleOdometry, "/fmu/in/vehicle_visual_odometry", PX4_QOS) if publish else None
        if publish:
            self.get_logger().warn("PUBLISHING to /fmu/in/vehicle_visual_odometry "
                                   "-- this feeds EKF2 and therefore control")

        self.create_subscription(SensorCombined, "/fmu/out/sensor_combined",
                                 self._on_imu, PX4_QOS)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude",
                                 self._on_att, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position",
                                 self._on_local, PX4_QOS)
        self.create_subscription(VehicleLocalPosition,
                                 "/fmu/out/vehicle_local_position_groundtruth",
                                 self._on_gt, PX4_QOS)

        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f"shadow mode: target {rate_hz:.0f} Hz for {duration:.0f}s")

    # -- subscriptions ----------------------------------------------------------
    def _on_imu(self, msg: SensorCombined) -> None:
        # PX4 stamps in microseconds since boot; pair it with a local arrival time so the
        # latency figure includes the transport, not just the compute.
        self.last_imu_wall = time.perf_counter()
        self.imu.append((msg.timestamp * 1e-6,
                         np.asarray(msg.gyro_rad, dtype=np.float32),
                         np.asarray(msg.accelerometer_m_s2, dtype=np.float32),
                         self.last_imu_wall))
        self._trim(self.imu)

    def _on_att(self, msg: VehicleAttitude) -> None:
        self.att.append((msg.timestamp * 1e-6, np.asarray(msg.q, dtype=np.float32)))
        self._trim(self.att)

    def _on_local(self, msg: VehicleLocalPosition) -> None:
        self.est_vel = [msg.vx, msg.vy, msg.vz]

    def _on_gt(self, msg: VehicleLocalPosition) -> None:
        self.gt = [msg.x, msg.y, msg.z]
        self.gt_vel = [msg.vx, msg.vy, msg.vz]

    def _now_us(self) -> int:
        """PX4 uORB stamps are microseconds on the same clock the samples arrive on."""
        return int(self.get_clock().now().nanoseconds / 1000)

    @staticmethod
    def _trim(buf: deque) -> None:
        cutoff = buf[-1][0] - BUFFER_SECONDS
        while len(buf) > 1 and buf[0][0] < cutoff:
            buf.popleft()

    # -- inference --------------------------------------------------------------
    def _tick(self) -> None:
        if time.perf_counter() - self.t_start > self.duration:
            self._finish()
            return
        if len(self.imu) < 2 or len(self.att) < 2:
            self.skipped += 1
            return
        span = min(self.imu[-1][0], self.att[-1][0]) - max(self.imu[0][0], self.att[0][0])
        if span < WINDOW_SECONDS:
            self.skipped += 1
            return

        imu = list(self.imu)
        att = list(self.att)
        newest_arrival = imu[-1][3]  # wall clock of the freshest sample in the window

        imu_t = np.array([s[0] for s in imu], dtype=np.float64)
        gyro = np.stack([s[1] for s in imu]).astype(np.float32)
        acc = np.stack([s[2] for s in imu]).astype(np.float32)
        att_t = np.array([s[0] for s in att], dtype=np.float64)
        quat = np.stack([s[1] for s in att]).astype(np.float32)

        payload = (REQ_MAGIC + struct.pack("<II", len(imu_t), len(att_t))
                   + imu_t.tobytes() + gyro.tobytes() + acc.tobytes()
                   + att_t.tobytes() + quat.tobytes())

        t_send = time.perf_counter()
        self.sock.send(payload)
        reply = self.sock.recv(1 << 16)
        t_done = time.perf_counter()

        if len(reply) < 5 or reply[:4] != REP_MAGIC:
            self.failed += 1
            return
        ok = reply[4]
        vel = np.frombuffer(reply, np.float32, 3, 5)
        disp = np.frombuffer(reply, np.float32, 3, 17)
        cov = np.frombuffer(reply, np.float32, 3, 29)
        t_ref, infer_ms = struct.unpack_from("<dd", reply, 41)
        if not ok:
            self.failed += 1
            return

        if self.pub_ev is not None:
            self._publish_ev(vel, cov, t_ref)

        self.records.append({
            "t": t_done - self.t_start,
            "latency_ms": (t_done - newest_arrival) * 1e3,   # spec: IMU 도착 -> 추정치 산출
            "roundtrip_ms": (t_done - t_send) * 1e3,
            "server_ms": infer_ms,
            "vel_ned": vel.tolist(),
            "disp_body": disp.tolist(),
            "cov": cov.tolist(),
            "ekf_vel_ned": list(self.est_vel),
            "gt_vel_ned": list(self.gt_vel),
            "gt_pos_ned": list(self.gt),
        })

    def _publish_ev(self, vel: np.ndarray, cov: np.ndarray, t_ref: float) -> None:
        """Feed EKF2 velocity only (결정 A/B): position stays NaN so the filter keeps its
        own position state and altitude stays on the barometer.

        The cov head is a variance in m^2 over one 0.09 s displacement step -- the loss is
        ``residual**2 / cov + log(cov)``, a Gaussian NLL -- so the velocity variance is
        cov / dt^2. Using the head for exactly this is what it was designed for.
        """
        m = VehicleOdometry()
        m.timestamp = self._now_us()
        m.timestamp_sample = int(t_ref * 1e6)  # same PX4 clock the samples arrived on
        m.pose_frame = VehicleOdometry.POSE_FRAME_NED
        m.position = [float("nan")] * 3
        m.q = [float("nan")] * 4
        m.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        m.velocity = [float(v) for v in vel]
        m.angular_velocity = [float("nan")] * 3
        m.position_variance = [float("nan")] * 3
        m.orientation_variance = [float("nan")] * 3
        m.velocity_variance = [float(max(c, 1e-6) / (STEP_DT ** 2)) for c in cov]
        m.reset_counter = 0
        m.quality = 0
        self.pub_ev.publish(m)

    # -- reporting --------------------------------------------------------------
    def _finish(self) -> None:
        self.timer.cancel()
        n = len(self.records)
        summary = {"inferences": n, "skipped": self.skipped, "failed": self.failed,
                   "target_rate_hz": self.rate_hz, "published_to_ekf2": self.publish_ev}

        if n >= 2:
            lat = np.array([r["latency_ms"] for r in self.records])
            elapsed = self.records[-1]["t"] - self.records[0]["t"]
            summary.update({
                "achieved_rate_hz": (n - 1) / elapsed if elapsed > 0 else 0.0,
                "latency_ms_mean": float(lat.mean()),
                "latency_ms_p50": float(np.percentile(lat, 50)),
                "latency_ms_p95": float(np.percentile(lat, 95)),
                "latency_ms_max": float(lat.max()),
                "server_ms_mean": float(np.mean([r["server_ms"] for r in self.records])),
                "roundtrip_ms_mean": float(np.mean([r["roundtrip_ms"] for r in self.records])),
            })
            # Velocity agreement against Gazebo truth, recorded for context only.
            pairs = [(r["vel_ned"], r["gt_vel_ned"]) for r in self.records
                     if not math.isnan(r["gt_vel_ned"][0])]
            if pairs:
                err = np.array([np.subtract(a, b) for a, b in pairs])
                summary["vel_rmse_vs_gt"] = float(np.sqrt((err ** 2).sum(1).mean()))
            summary["passed"] = (summary["achieved_rate_hz"] >= 20.0
                                 and summary["latency_ms_p95"] <= 50.0)
            for k in ("achieved_rate_hz", "latency_ms_p50", "latency_ms_p95",
                      "latency_ms_max", "server_ms_mean", "vel_rmse_vs_gt"):
                if k in summary:
                    self.get_logger().info(f"{k}: {summary[k]:.3f}")
        else:
            summary["passed"] = False
            self.get_logger().error("not enough inferences to score")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(
            json.dumps({"summary": summary, "records": self.records}, indent=2),
            encoding="utf-8")
        self.get_logger().info(f"log: {self.out_path}")
        self.sock.close()
        self.finished = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/mario_infer.sock")
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "stage2_shadow.json")
    ap.add_argument("--publish", action="store_true",
                    help="Stage 5: feed EKF2 via /fmu/in/vehicle_visual_odometry")
    args = ap.parse_args()

    rclpy.init()
    node = MarioOdometryNode(args.socket, args.rate, args.duration, args.out,
                         publish=args.publish)
    # spin_once in a loop rather than rclpy.spin(): calling rclpy.shutdown() from inside a
    # timer callback left spin() blocked and the process alive after the log was written,
    # so finished runs piled up as zombies still streaming setpoints.
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
