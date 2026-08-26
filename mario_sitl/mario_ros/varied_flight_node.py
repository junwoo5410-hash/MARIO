#!/usr/bin/env python
"""Stage 4b — fly varied manoeuvres to collect the regime Blackbird never covered.

Stage 4 showed MARIO emits a 0.7-1.1 m/s phantom velocity when the airframe is still,
because across all ten Blackbird eval trajectories the minimum speed is 0.178 m/s and only
0.03% of samples sit below 0.2 m/s. The network simply never saw a slow or stationary
vehicle. This node deliberately spends most of its time there, while still touching the
fast regime so fine-tuning does not trade one bias for another.

Speed is commanded by walking the position setpoint at a chosen rate rather than by
jumping it to the target, which would just make PX4 fly at its own maximum.

Ground truth is recorded by record_flight_node.py, never used here for control
(spec §1: "Gazebo GT: 검증용으로만 기록, 제어에 사용 금지").
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, \
    VehicleLocalPosition, VehicleStatus
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
ORIGIN_SETTLE_SECONDS = 3.0

# The regime the training set is missing sits at the bottom of this list; the fast entries
# keep the fine-tune anchored to what the network already does well.
SPEEDS = [0.15, 0.25, 0.4, 0.7, 1.0, 1.5, 2.5, 3.5]
BOX_XY = 25.0          # stay inside +-25 m of the launch point
ALT_MIN, ALT_MAX = 2.0, 14.0


class VariedFlightNode(Node):
    def __init__(self, duration: float, seed: int):
        super().__init__("varied_flight_node")
        self.duration = duration
        self.rng = random.Random(seed)
        self.finished = False

        self.pub_mode = self.create_publisher(OffboardControlMode,
                                              "/fmu/in/offboard_control_mode", PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint,
                                            "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position",
                                 self._on_local, PX4_QOS)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status",
                                 self._on_status, PX4_QOS)

        self.est = [math.nan] * 3
        self.est_valid = False
        self.status = None
        self.origin = None
        self.valid_since = None

        self.ticks = 0
        self.phase = "primer"
        self.ref = None          # the setpoint we walk around
        self.target = None
        self.speed = 1.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.hold_until = 0.0
        self.t_start = time.perf_counter()
        self.plan_log = []

        self.timer = self.create_timer(1.0 / HZ, self._tick)
        self.get_logger().info(f"varied flight for {duration:.0f}s, seed {seed}")

    def _on_local(self, m: VehicleLocalPosition) -> None:
        self.est = [m.x, m.y, m.z]
        self.est_valid = bool(m.xy_valid and m.z_valid)

    def _on_status(self, m: VehicleStatus) -> None:
        self.status = m

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _cmd(self, command: int, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        for i in range(1, 8):
            setattr(msg, f"param{i}", float(params.get(f"param{i}", 0.0)))
        msg.target_system = msg.target_component = 1
        msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._now_us()
        self.pub_cmd.publish(msg)

    def _publish(self, pos) -> None:
        mode = OffboardControlMode()
        mode.position = True
        mode.timestamp = self._now_us()
        self.pub_mode.publish(mode)
        sp = TrajectorySetpoint()
        sp.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        sp.yaw = float(self.yaw)
        sp.timestamp = self._now_us()
        self.pub_sp.publish(sp)

    # -- manoeuvre selection ----------------------------------------------------
    def _next_manoeuvre(self) -> None:
        """Pick the next thing to do, weighted towards the under-represented regime."""
        r = self.rng.random()
        base = self.origin
        if r < 0.30:
            # hold still: the case the training set has essentially none of
            self.target = list(self.ref)
            self.speed = 0.0
            self.hold_until = time.perf_counter() + self.rng.uniform(4.0, 12.0)
            kind = "hover"
        elif r < 0.45:
            # pure vertical, slow: also absent from Blackbird
            dz = self.rng.uniform(-3.0, 3.0)
            z = min(max(self.ref[2] + dz, base[2] - ALT_MAX), base[2] - ALT_MIN)
            self.target = [self.ref[0], self.ref[1], z]
            self.speed = self.rng.choice(SPEEDS[:4])
            self.hold_until = 0.0
            kind = "climb"
        else:
            ang = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(1.0, 14.0)
            x = min(max(self.ref[0] + dist * math.cos(ang), base[0] - BOX_XY), base[0] + BOX_XY)
            y = min(max(self.ref[1] + dist * math.sin(ang), base[1] - BOX_XY), base[1] + BOX_XY)
            z = min(max(self.ref[2] + self.rng.uniform(-2.0, 2.0),
                        base[2] - ALT_MAX), base[2] - ALT_MIN)
            self.target = [x, y, z]
            self.speed = self.rng.choice(SPEEDS)
            self.hold_until = 0.0
            kind = "goto"

        # keep yaw moving so the body frame is exercised, not just translation
        self.yaw_rate = self.rng.uniform(-0.35, 0.35)
        self.plan_log.append({"t": time.perf_counter() - self.t_start, "kind": kind,
                              "speed": self.speed, "target": list(self.target)})
        self.get_logger().info(f"{kind:<6} speed {self.speed:>4.2f} m/s "
                               f"-> ({self.target[0]:+.1f}, {self.target[1]:+.1f}, "
                               f"{self.target[2]:+.1f})")

    def _walk_ref(self) -> None:
        """Step the reference toward the target at the commanded speed."""
        self.yaw = (self.yaw + self.yaw_rate / HZ + math.pi) % (2 * math.pi) - math.pi
        if self.speed <= 0.0:
            return
        d = [self.target[i] - self.ref[i] for i in range(3)]
        dist = math.sqrt(sum(v * v for v in d))
        step = self.speed / HZ
        if dist <= step:
            self.ref = list(self.target)
            return
        for i in range(3):
            self.ref[i] += d[i] * step / dist

    def _arrived(self) -> bool:
        if self.hold_until:
            return time.perf_counter() >= self.hold_until
        return all(abs(self.ref[i] - self.target[i]) < 1e-6 for i in range(3))

    # -- state machine ----------------------------------------------------------
    def _tick(self) -> None:
        self.ticks += 1
        if self.origin is None:
            self._latch()

        if self.phase == "primer":
            hover = ([0.0, 0.0, -4.0] if self.origin is None
                     else [self.origin[0], self.origin[1], self.origin[2] - 4.0])
            self._publish(hover)
            if self.ticks >= PRIMER_TICKS and self.origin is not None:
                self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
                self.ref = list(hover)
                self.target = list(hover)
                self.speed = 1.0
                self.phase = "climb_out"
                self.phase_t = time.perf_counter()
            return

        if self.phase == "climb_out":
            self._publish(self.ref)
            if time.perf_counter() - self.phase_t > 12.0:
                self.phase = "fly"
                self._next_manoeuvre()
            return

        if self.phase == "fly":
            self._walk_ref()
            self._publish(self.ref)
            if time.perf_counter() - self.t_start > self.duration:
                self.phase = "land"
                self._cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.phase_t = time.perf_counter()
                self.get_logger().info("landing")
                return
            if self._arrived():
                self._next_manoeuvre()
            return

        if self.phase == "land":
            disarmed = (self.status is not None
                        and self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED)
            if disarmed or time.perf_counter() - self.phase_t > 60.0:
                self.timer.cancel()
                self.get_logger().info(f"done, {len(self.plan_log)} manoeuvres")
                self.finished = True
            return

    def _latch(self) -> None:
        now = self.get_clock().now()
        if not self.est_valid or math.isnan(self.est[0]):
            self.valid_since = None
            return
        if self.valid_since is None:
            self.valid_since = now
            return
        if (now - self.valid_since).nanoseconds / 1e9 >= ORIGIN_SETTLE_SECONDS:
            self.origin = list(self.est)
            self.get_logger().info(f"origin latched {[round(v, 2) for v in self.origin]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rclpy.init()
    node = VariedFlightNode(args.duration, args.seed)
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
