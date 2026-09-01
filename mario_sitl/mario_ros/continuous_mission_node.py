#!/usr/bin/env python
"""Closed-loop flight that never asks the airframe to hold position.

Every Stage 5/6 run failed at the same gate: the mission required the estimate to settle
within 0.3 m of a fixed setpoint, and an IMU-only estimator cannot do that because velocity
is unobservable near zero speed (drag signal = k*v, and it vanishes with v). All ten sweep
runs died in takeoff without ever exercising their variables.

This flies a circle instead. The setpoint moves continuously at a speed well above the
0.23 m/s observability floor, in the regime where MARIO measurably works (TDE 1.9-7.3%),
and there is no settle gate anywhere.

GPS is used for takeoff and one reference lap, then cut IN FLIGHT. That is both the only
way past the takeoff problem and the realistic scenario: GPS is lost during flight -- a
tunnel, an urban canyon, jamming -- not absent at power-on. EKF2_GPS_CTRL is not
reboot_required, so it can be switched at runtime.

A closed path makes the drift self-evident: ground truth returns to where it started, so
any gap between laps is estimator error, not mission geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleLocalPosition, VehicleStatus)
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)

HZ = 50.0
PRIMER_TICKS = 20
ORIGIN_SETTLE_S = 3.0
PX4_PARAM = "/src/gs25122/PX4-Autopilot/build/px4_sitl_default/bin/px4-param"
PX4_ROOTFS = "/src/gs25122/PX4-Autopilot/build/px4_sitl_default/rootfs"


class ContinuousMissionNode(Node):
    def __init__(self, alt: float, radius: float, speed: float,
                 laps_gps: float, laps_mario: float, out: Path):
        super().__init__("continuous_mission_node")
        self.alt, self.radius, self.speed = alt, radius, speed
        self.laps_gps, self.laps_mario = laps_gps, laps_mario
        self.out = out
        self.finished = False

        self.pub_mode = self.create_publisher(OffboardControlMode,
                                              "/fmu/in/offboard_control_mode", PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint,
                                            "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position",
                                 self._on_local, PX4_QOS)
        self.create_subscription(VehicleLocalPosition,
                                 "/fmu/out/vehicle_local_position_groundtruth",
                                 self._on_gt, PX4_QOS)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status",
                                 self._on_status, PX4_QOS)

        self.est = [math.nan] * 3
        self.est_valid = False
        self.gt = [math.nan] * 3
        self.status = None
        self.origin = None
        self.gt_origin = None
        self.valid_since = None

        self.ticks = 0
        self.phase = "primer"
        self.phase_t = time.perf_counter()
        self.theta = 0.0          # angle around the circle
        self.gps_cut_t = None
        self.gps_cut_gt = None
        self.samples: list = []
        self.events: list = []

        self.timer = self.create_timer(1.0 / HZ, self._tick)
        lap = 2 * math.pi * radius / speed
        self.get_logger().info(
            f"circle r={radius} m at {speed} m/s (lap {lap:.1f} s): "
            f"{laps_gps} lap(s) on GPS, then GPS CUT, then {laps_mario} lap(s) on MARIO")

    # -- subscriptions ---------------------------------------------------------
    def _on_local(self, m: VehicleLocalPosition) -> None:
        self.est = [m.x, m.y, m.z]
        self.est_valid = bool(m.xy_valid and m.z_valid)

    def _on_gt(self, m: VehicleLocalPosition) -> None:
        self.gt = [m.x, m.y, m.z]

    def _on_status(self, m: VehicleStatus) -> None:
        self.status = m

    # -- helpers ---------------------------------------------------------------
    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _cmd(self, command: int, **p) -> None:
        msg = VehicleCommand()
        msg.command = command
        for i in range(1, 8):
            setattr(msg, f"param{i}", float(p.get(f"param{i}", 0.0)))
        msg.target_system = msg.target_component = 1
        msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._now_us()
        self.pub_cmd.publish(msg)

    def _publish(self, pos, yaw: float = 0.0) -> None:
        mode = OffboardControlMode()
        mode.position = True
        mode.timestamp = self._now_us()
        self.pub_mode.publish(mode)
        sp = TrajectorySetpoint()
        sp.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        sp.yaw = float(yaw)
        sp.timestamp = self._now_us()
        self.pub_sp.publish(sp)
        self.samples.append({"t": time.perf_counter(), "phase": self.phase,
                             "setpoint": list(pos), "est": list(self.est),
                             "gt": list(self.gt)})

    def _circle_point(self, theta: float):
        """Centre the circle so that theta=0 sits at the takeoff point."""
        b = self.origin
        cx, cy = b[0], b[1] + self.radius
        return [cx + self.radius * math.sin(theta),
                cy - self.radius * math.cos(theta),
                b[2] - self.alt]

    def _event(self, name: str) -> None:
        self.events.append({"t": time.perf_counter(), "name": name,
                            "gt": list(self.gt), "est": list(self.est),
                            "theta": self.theta})
        self.get_logger().info(f"{name}  gt=({self.gt[0]:+.2f}, {self.gt[1]:+.2f}, "
                               f"{self.gt[2]:+.2f})")

    def _cut_gps(self) -> None:
        """Disable GNSS aiding at runtime. EKF2_GPS_CTRL is not reboot_required."""
        subprocess.run([PX4_PARAM, "set", "EKF2_GPS_CTRL", "0"],
                       cwd=PX4_ROOTFS, capture_output=True, timeout=20)
        self.gps_cut_t = time.perf_counter()
        self.gps_cut_gt = list(self.gt)
        self._event("GPS CUT — MARIO only from here")

    # -- state machine ---------------------------------------------------------
    def _tick(self) -> None:
        self.ticks += 1
        if self.origin is None:
            self._latch()

        if self.phase == "primer":
            base = [0.0, 0.0, -self.alt] if self.origin is None else \
                   [self.origin[0], self.origin[1], self.origin[2] - self.alt]
            self._publish(base)
            if self.ticks >= PRIMER_TICKS and self.origin is not None:
                self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
                self.phase, self.phase_t = "climb", time.perf_counter()
                self._event("armed, climbing on GPS")
            return

        if self.phase == "climb":
            self._publish(self._circle_point(0.0))
            if time.perf_counter() - self.phase_t > 15.0:
                self.phase, self.phase_t = "lap_gps", time.perf_counter()
                self._event("circling on GPS (reference lap)")
            return

        if self.phase in ("lap_gps", "lap_mario"):
            self.theta += (self.speed / self.radius) / HZ
            # face along the direction of travel, so the body frame is exercised
            self._publish(self._circle_point(self.theta), yaw=self.theta)
            laps = self.theta / (2 * math.pi)
            if self.phase == "lap_gps" and laps >= self.laps_gps:
                self._cut_gps()
                self.phase, self.phase_t = "lap_mario", time.perf_counter()
            elif self.phase == "lap_mario" and laps >= self.laps_gps + self.laps_mario:
                self.phase, self.phase_t = "land", time.perf_counter()
                self._cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._event("landing")
            return

        if self.phase == "land":
            disarmed = (self.status is not None
                        and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED)
            if disarmed or time.perf_counter() - self.phase_t > 45.0:
                self._finish()
            return

    def _latch(self) -> None:
        now = self.get_clock().now()
        if not self.est_valid or math.isnan(self.est[0]) or math.isnan(self.gt[0]):
            self.valid_since = None
            return
        if self.valid_since is None:
            self.valid_since = now
            return
        if (now - self.valid_since).nanoseconds / 1e9 >= ORIGIN_SETTLE_S:
            self.origin = list(self.est)
            self.gt_origin = list(self.gt)
            self.get_logger().info(f"origin latched est={[round(v,2) for v in self.origin]} "
                                   f"gt={[round(v,2) for v in self.gt_origin]}")

    # -- reporting -------------------------------------------------------------
    def _finish(self) -> None:
        self.timer.cancel()
        s = {"radius": self.radius, "speed": self.speed, "alt": self.alt,
             "laps_gps": self.laps_gps, "laps_mario": self.laps_mario,
             "ekf_origin": self.origin, "gt_origin": self.gt_origin,
             "gps_cut_gt": self.gps_cut_gt}

        rows = [x for x in self.samples
                if not (np.isnan(np.asarray(x["gt"], dtype=float)).any()
                        or np.isnan(np.asarray(x["est"], dtype=float)).any())]
        after = [x for x in rows if x["phase"] == "lap_mario"]
        if after and self.gt_origin is not None:
            gt = np.array([x["gt"] for x in after], dtype=float)
            est = np.array([x["est"] for x in after], dtype=float)
            sp = np.array([x["setpoint"] for x in after], dtype=float)
            t = np.array([x["t"] for x in after]); t -= t[0]
            gt_d = gt - np.array(self.gt_origin)
            est_d = est - np.array(self.origin)
            sp_d = sp - np.array(self.origin)
            err = np.linalg.norm(est_d - gt_d, axis=1)
            dev = np.linalg.norm(gt_d - sp_d, axis=1)
            dist = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
            s.update({
                "mario_only_seconds": float(t[-1]),
                "distance_flown_m": dist,
                "est_error_final_m": float(err[-1]),
                "est_error_max_m": float(err.max()),
                "est_error_pct_of_distance": float(100 * err[-1] / max(dist, 1e-6)),
                "vehicle_deviation_final_m": float(dev[-1]),
                "vehicle_deviation_max_m": float(dev.max()),
                "drift_rate_m_per_s": float(np.polyfit(t, err, 1)[0]),
            })
            for k in ("mario_only_seconds", "distance_flown_m", "est_error_final_m",
                      "est_error_pct_of_distance", "vehicle_deviation_max_m",
                      "drift_rate_m_per_s"):
                self.get_logger().info(f"{k}: {s[k]:.3f}")

        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps({"summary": s, "events": self.events,
                                        "samples": self.samples}, indent=2))
        self.get_logger().info(f"log: {self.out}")
        self.finished = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=6.0)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=3.5)
    ap.add_argument("--laps-gps", type=float, default=1.0)
    ap.add_argument("--laps-mario", type=float, default=3.0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "continuous_run.json")
    a = ap.parse_args()

    rclpy.init()
    node = ContinuousMissionNode(a.alt, a.radius, a.speed, a.laps_gps, a.laps_mario, a.out)
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
