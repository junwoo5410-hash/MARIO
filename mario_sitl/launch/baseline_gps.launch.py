#!/usr/bin/env python
"""Stage 1 — GPS-enabled baseline: agent + headless SITL + offboard mission.

    ros2 launch mario_sitl/launch/baseline_gps.launch.py

Brings up the three processes the closed-loop study needs, in the only order that works:

  1. MicroXRCEAgent   — must already be listening when PX4 boots, otherwise the
                        uXRCE-DDS client retries and the first seconds of topics are lost.
  2. PX4 SITL + gz    — HEADLESS=1 because this host has no display. That switches gz to
                        server-only; physics and the groundtruth topics are unaffected.
  3. offboard node    — delayed until PX4 has published its first topics.

Paths are absolute on purpose: everything was built without root under /src.
"""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration

PX4_DIR = Path("/src/gs25122/PX4-Autopilot")
PX4_ENV = Path("/src/gs25122/miniconda3/envs/px4")
AGENT_BIN = Path("/src/gs25122/xrce_agent/bin/MicroXRCEAgent")
NODE = Path(__file__).resolve().parents[1] / "mario_ros" / "offboard_mission_node.py"

#: PX4 boots, EKF2 converges and the DDS session establishes well inside this window.
NODE_START_DELAY = 20.0


def _px4_environment() -> dict:
    """PX4 build used the conda px4 env for cmake and Gazebo; the run needs the same libs."""
    env = dict(os.environ)
    env["PATH"] = f"{PX4_ENV / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{PX4_ENV / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["HEADLESS"] = "1"  # no display on this host
    env["PX4_GZ_MODEL_POSE"] = "0,0,0,0,0,0"
    return env


def generate_launch_description() -> LaunchDescription:
    hover = LaunchConfiguration("hover")
    north = LaunchConfiguration("north")
    out = LaunchConfiguration("out")

    agent = ExecuteProcess(
        cmd=[str(AGENT_BIN), "udp4", "-p", "8888"],
        name="micro_xrce_agent",
        output="screen",
    )

    sitl = ExecuteProcess(
        cmd=["make", "px4_sitl", "gz_x500"],
        cwd=str(PX4_DIR),
        name="px4_sitl",
        output="screen",
        additional_env=_px4_environment(),
    )

    mission = TimerAction(
        period=NODE_START_DELAY,
        actions=[
            ExecuteProcess(
                cmd=["python", str(NODE), "--hover", hover, "--north", north, "--out", out],
                name="offboard_mission",
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("hover", default_value="30.0"),
        DeclareLaunchArgument("north", default_value="5.0"),
        DeclareLaunchArgument(
            "out",
            default_value=str(Path(__file__).resolve().parents[1] / "results" / "stage1_baseline.json"),
        ),
        agent,
        sitl,
        mission,
    ])
