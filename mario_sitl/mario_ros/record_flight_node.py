#!/usr/bin/env python
"""Stage 4 support — record a raw SITL flight for offline MARIO evaluation.

The shadow node keeps only per-inference outputs, but ``mario.evaluate.rollout_trajectory``
needs the whole stream. This node dumps the raw PX4 topics to an ``.npz``; all resampling
and frame conversion happens offline in ``scripts/eval_openloop.py``, so nothing about the
evaluation depends on this node's timing.

Both attitude sources are recorded on purpose. Training fed the network ground-truth
attitude (spec 함정 C) but closed-loop can only ever supply the EKF estimate, so Stage 4
scores each separately and the difference is the cost of that mismatch.

numpy-only: this runs in the py3.11 ROS env, which has no torch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import SensorCombined, VehicleAttitude, VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


class RecordFlightNode(Node):
    def __init__(self, out_path: Path, duration: float):
        super().__init__("record_flight_node")
        self.out_path = out_path
        self.duration = duration
        self.finished = False
        self.t0 = time.perf_counter()

        self.imu: list = []
        self.att_ekf: list = []
        self.att_gt: list = []
        self.pos_gt: list = []
        self.pos_ekf: list = []

        self.create_subscription(SensorCombined, "/fmu/out/sensor_combined",
                                 self._on_imu, PX4_QOS)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude",
                                 self._on_att_ekf, PX4_QOS)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude_groundtruth",
                                 self._on_att_gt, PX4_QOS)
        self.create_subscription(VehicleLocalPosition,
                                 "/fmu/out/vehicle_local_position_groundtruth",
                                 self._on_pos_gt, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position",
                                 self._on_pos_ekf, PX4_QOS)

        self.timer = self.create_timer(0.5, self._tick)
        self.get_logger().info(f"recording {duration:.0f}s -> {out_path}")

    def _on_imu(self, m: SensorCombined) -> None:
        self.imu.append((m.timestamp * 1e-6, *m.gyro_rad, *m.accelerometer_m_s2))

    def _on_att_ekf(self, m: VehicleAttitude) -> None:
        self.att_ekf.append((m.timestamp * 1e-6, *m.q))

    def _on_att_gt(self, m: VehicleAttitude) -> None:
        self.att_gt.append((m.timestamp * 1e-6, *m.q))

    def _on_pos_gt(self, m: VehicleLocalPosition) -> None:
        self.pos_gt.append((m.timestamp * 1e-6, m.x, m.y, m.z, m.vx, m.vy, m.vz))

    def _on_pos_ekf(self, m: VehicleLocalPosition) -> None:
        self.pos_ekf.append((m.timestamp * 1e-6, m.x, m.y, m.z, m.vx, m.vy, m.vz))

    def _tick(self) -> None:
        if time.perf_counter() - self.t0 < self.duration:
            return
        self.timer.cancel()
        arrays = {k: np.array(v, dtype=np.float64) for k, v in
                  {"imu": self.imu, "att_ekf": self.att_ekf, "att_gt": self.att_gt,
                   "pos_gt": self.pos_gt, "pos_ekf": self.pos_ekf}.items()}
        for k, a in arrays.items():
            self.get_logger().info(f"{k}: {len(a)} samples")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out_path, **arrays)
        self.get_logger().info(f"wrote {self.out_path}")
        self.finished = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "stage4_flight.npz")
    args = ap.parse_args()

    rclpy.init()
    node = RecordFlightNode(args.out, args.duration)
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
