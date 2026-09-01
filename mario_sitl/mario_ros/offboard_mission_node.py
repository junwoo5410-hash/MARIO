#!/usr/bin/env python
"""Stage 1 — offboard control skeleton, MARIO not involved.

Flies the spec's reference mission with GPS enabled so that later closed-loop failures
can be attributed to the estimator rather than to this control code:

    arm -> climb to 3 m -> 5 m north -> hover 30 s -> land

Success criterion (spec Stage 1): reach the target within 0.3 m of Gazebo ground truth.

PX4 conventions that this node depends on, all of them load-bearing:
  * NED world frame, so "3 m altitude" is z = -3.0 and "5 m north" is +x.
  * Offboard mode is rejected unless setpoints are already streaming: PX4 requires a
    prior stream before the mode switch, and drops out of offboard if the stream falls
    below 2 Hz. We pre-stream SETPOINT_PRIMER ticks and then hold 50 Hz throughout.
  * uORB timestamps are microseconds.

Ground truth is read from /fmu/out/vehicle_local_position_groundtruth, which PX4 SITL
publishes from the simulator state. It is recorded for scoring only and never feeds a
setpoint (spec §1: "Gazebo GT: 검증용으로만 기록, 제어에 사용 금지").
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

SETPOINT_HZ = 50.0  # spec: 최소 50 Hz, 2 Hz 미만이면 페일세이프
SETPOINT_PRIMER = 20  # spec: 10회 이상 선발행. 20 ticks @50 Hz = 0.4 s
TAKEOFF_ALT = 3.0  # metres above the arming point
NORTH_DISTANCE = 5.0
HOVER_SECONDS = 30.0
ARRIVAL_RADIUS = 0.3  # spec Stage 1 성공 기준
SETTLE_SECONDS = 2.0  # time inside the radius before a phase is considered reached
ORIGIN_SETTLE_SECONDS = 3.0  # estimator must hold xy_valid+z_valid this long before we latch
LAND_TIMEOUT = 60.0  # give up waiting for disarm

# PX4 uORB best-effort publishers need a matching subscriber profile.
PX4_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


@dataclass
class Sample:
    t: float
    phase: str
    setpoint: list
    est: list
    gt: list


@dataclass
class MissionLog:
    samples: list = field(default_factory=list)
    events: list = field(default_factory=list)


class OffboardMissionNode(Node):
    def __init__(self, out_path: Path, hover_seconds: float, north: float, alt: float):
        super().__init__("offboard_mission_node")
        self.out_path = out_path
        self.hover_seconds = hover_seconds
        self.north = north
        self.alt = alt

        self.pub_mode = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self.pub_sp = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)

        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status", self._on_status, PX4_QOS)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_local, PX4_QOS)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_groundtruth",
            self._on_gt, PX4_QOS)

        self.status: VehicleStatus | None = None
        self.est = [math.nan] * 3
        self.est_valid = False
        self.gt = [math.nan] * 3
        self.origin: list | None = None
        self.gt_origin: list | None = None
        self.valid_since: rclpy.time.Time | None = None
        self.land_sent = False

        self.finished = False
        self.cruise_speed = 0.0   # 0 = step the setpoint (PX4 picks its own profile)
        self.cruise_ref = None
        self.ticks = 0
        self.phase = "primer"
        self.phase_start = self.get_clock().now()
        self.inside_since: rclpy.time.Time | None = None
        self.log = MissionLog()

        self.timer = self.create_timer(1.0 / SETPOINT_HZ, self._tick)
        self.get_logger().info(
            f"mission: arm -> {alt:.1f} m -> {north:.1f} m north -> hover {hover_seconds:.0f}s -> land")

    # -- subscriptions ----------------------------------------------------------
    def _on_status(self, msg: VehicleStatus) -> None:
        self.status = msg

    def _on_local(self, msg: VehicleLocalPosition) -> None:
        self.est = [msg.x, msg.y, msg.z]
        self.est_valid = bool(msg.xy_valid and msg.z_valid)

    def _on_gt(self, msg: VehicleLocalPosition) -> None:
        self.gt = [msg.x, msg.y, msg.z]

    # -- helpers ----------------------------------------------------------------
    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_start).nanoseconds / 1e9

    def _enter(self, phase: str) -> None:
        self.get_logger().info(f"phase -> {phase} (gt={self._fmt(self.gt)})")
        self.log.events.append({"t": self._now_us() / 1e6, "phase": phase,
                                "gt": list(self.gt), "est": list(self.est)})
        self.phase = phase
        self.phase_start = self.get_clock().now()
        self.inside_since = None

    @staticmethod
    def _fmt(v) -> str:
        return "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"

    def _send_command(self, command: int, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        for i in range(1, 8):
            setattr(msg, f"param{i}", float(params.get(f"param{i}", 0.0)))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._now_us()
        self.pub_cmd.publish(msg)

    def _publish_setpoint(self, target: list) -> None:
        mode = OffboardControlMode()
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        mode.timestamp = self._now_us()
        self.pub_mode.publish(mode)

        sp = TrajectorySetpoint()
        sp.position = [float(target[0]), float(target[1]), float(target[2])]
        sp.yaw = 0.0  # hold north-facing; keeps the body frame aligned with NED
        sp.timestamp = self._now_us()
        self.pub_sp.publish(sp)

        self.log.samples.append(Sample(self._now_us() / 1e6, self.phase,
                                       list(target), list(self.est), list(self.gt)).__dict__)

    def _reached(self, target: list) -> bool:
        """True once the estimate has stayed inside ARRIVAL_RADIUS for SETTLE_SECONDS."""
        if math.isnan(self.est[0]):
            return False
        d = math.dist(self.est, target)
        now = self.get_clock().now()
        if d > ARRIVAL_RADIUS:
            self.inside_since = None
            return False
        if self.inside_since is None:
            self.inside_since = now
        return (now - self.inside_since).nanoseconds / 1e9 >= SETTLE_SECONDS

    def _try_latch_origin(self) -> None:
        """Latch the mission base only after the estimator declares itself valid and holds
        that for ORIGIN_SETTLE_SECONDS.

        The first Stage 1 run latched on the first non-NaN estimate, capturing an EKF that
        was still 0.82 m off in z while the airframe sat on the ground. That offset then
        biased every setpoint for the whole flight, so the vehicle flew 3.45 m up on a
        3.0 m command. Both the EKF and the GT base are recorded at the same instant so
        the two frames can be compared honestly afterwards.
        """
        now = self.get_clock().now()
        if not self.est_valid or math.isnan(self.est[0]) or math.isnan(self.gt[0]):
            self.valid_since = None
            return
        if self.valid_since is None:
            self.valid_since = now
            return
        if (now - self.valid_since).nanoseconds / 1e9 < ORIGIN_SETTLE_SECONDS:
            return
        self.origin = list(self.est)
        self.gt_origin = list(self.gt)
        self.get_logger().info(
            f"origin latched est={self._fmt(self.origin)} gt={self._fmt(self.gt_origin)}")

    def _disarmed(self) -> bool:
        return (self.status is not None
                and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED)

    # -- state machine ----------------------------------------------------------
    def _tick(self) -> None:
        self.ticks += 1
        if self.origin is None:
            self._try_latch_origin()

        base = self.origin or [0.0, 0.0, 0.0]
        hover = [base[0], base[1], base[2] - self.alt]  # NED: up is -z
        goal = [base[0] + self.north, base[1], base[2] - self.alt]

        if self.phase == "primer":
            # Stream setpoints BEFORE requesting offboard, else PX4 rejects the switch.
            self._publish_setpoint(hover)
            if self.ticks >= SETPOINT_PRIMER and self.origin is not None:
                self._send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                                   param1=1.0, param2=6.0)  # 6 = PX4 offboard
                self._send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
                self._enter("takeoff")
            return

        if self.phase == "takeoff":
            self._publish_setpoint(hover)
            if self._reached(hover):
                self._enter("cruise")
            elif self._elapsed() > 30.0:
                self.get_logger().error("takeoff timed out")
                self._enter("land")
            return

        if self.phase == "cruise":
            # A commanded speed walks the setpoint, producing near-constant-velocity
            # flight; Stage 4c showed that regime carries almost no velocity information
            # in the IMU, so this axis probes observability with the loop closed.
            if self.cruise_speed > 0.0:
                if self.cruise_ref is None:
                    self.cruise_ref = list(hover)
                d = [goal[i] - self.cruise_ref[i] for i in range(3)]
                dist = math.sqrt(sum(v * v for v in d))
                step = self.cruise_speed / SETPOINT_HZ
                if dist <= step:
                    self.cruise_ref = list(goal)
                else:
                    for i in range(3):
                        self.cruise_ref[i] += d[i] * step / dist
                self._publish_setpoint(self.cruise_ref)
            else:
                self._publish_setpoint(goal)
            if self._reached(goal):
                self._enter("hover")
            elif self._elapsed() > 60.0:
                self.get_logger().error("cruise timed out")
                self._enter("land")
            return

        if self.phase == "hover":
            self._publish_setpoint(goal)
            if self._elapsed() >= self.hover_seconds:
                self._enter("land")
            return

        if self.phase == "land":
            if not self.land_sent:
                self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.land_sent = True
            # Wait for the actual touchdown. Run 1 stepped straight to "done" in one tick,
            # so its "final" error was sampled at the top of the descent and meant nothing.
            if self._disarmed() or self._elapsed() > LAND_TIMEOUT:
                self._enter("done")
            return

        if self.phase == "done":
            self._finish(goal)

    def _finish(self, goal: list) -> None:
        self.timer.cancel()
        hover = [s for s in self.log.samples if s["phase"] == "hover"]
        gt_hover = [s["gt"] for s in hover if not math.isnan(s["gt"][0])]

        est_hover = [s["est"] for s in hover if not math.isnan(s["est"][0])]
        commanded = [self.north, 0.0, -self.alt]

        summary = {
            "goal_ned_ekf": goal,
            "commanded_disp_ned": commanded,
            "ekf_origin": self.origin,
            "gt_origin": self.gt_origin,
            "arrival_radius": ARRIVAL_RADIUS,
        }
        if gt_hover and self.gt_origin is not None:
            # Score displacement against displacement. `goal` lives in the EKF local frame,
            # whose origin is offset from the simulator origin by the estimator's own error;
            # run 1 compared GT positions against that goal directly and so charged the
            # mission for a frame offset it never committed.
            def disp(p, ref):
                return [p[i] - ref[i] for i in range(3)]

            arrival = disp(gt_hover[0], self.gt_origin)
            final = disp(gt_hover[-1], self.gt_origin)
            summary["gt_arrival_disp"] = arrival
            summary["gt_final_disp"] = final
            summary["gt_arrival_error"] = math.dist(arrival, commanded)
            summary["gt_arrival_error_xy"] = math.dist(arrival[:2], commanded[:2])
            summary["gt_arrival_error_z"] = abs(arrival[2] - commanded[2])
            summary["gt_final_error"] = math.dist(final, commanded)
            summary["gt_hover_drift"] = math.dist(gt_hover[0], gt_hover[-1])
            if est_hover:
                # How well the controller tracked the setpoint in the frame it actually
                # flies in. Separates control error from estimator error.
                summary["ekf_tracking_error"] = math.dist(est_hover[0], goal)
            summary["passed"] = summary["gt_arrival_error"] < ARRIVAL_RADIUS
            for key in ("gt_arrival_error", "gt_arrival_error_xy", "gt_arrival_error_z",
                        "gt_final_error", "gt_hover_drift", "ekf_tracking_error"):
                if key in summary:
                    self.get_logger().info(f"{key}: {summary[key]:.3f} m")
        else:
            summary["passed"] = False
            self.get_logger().error("no ground-truth samples during hover")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(
            {"summary": summary, "events": self.log.events, "samples": self.log.samples},
            indent=2), encoding="utf-8")
        self.get_logger().info(f"log: {self.out_path}")
        self.finished = True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "results" / "stage1_baseline.json")
    p.add_argument("--hover", type=float, default=HOVER_SECONDS)
    p.add_argument("--north", type=float, default=NORTH_DISTANCE)
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    p.add_argument("--cruise-speed", type=float, default=0.0,
                   help="m/s; 0 steps the setpoint and lets PX4 choose the profile")
    args = p.parse_args()

    rclpy.init()
    node = OffboardMissionNode(args.out, args.hover, args.north, args.alt)
    node.cruise_speed = args.cruise_speed
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
