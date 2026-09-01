#!/usr/bin/env python
"""Generate a Gazebo + PX4 airframe from measured quadrotor numbers.

PX4 ships no TBS Discovery model, so this builds one. Everything that can be derived from
geometry is derived rather than copied: inertia comes from motors treated as point masses
at their real radius plus a box body, and the thrust constant is scaled to hold the same
thrust-to-weight ratio the stock x500 flies at.

Anything NOT derivable from the three inputs is inherited from x500 and printed as such, so
the estimates are visible instead of buried. Correct them with the flags and regenerate.

    make_airframe.py --name tbs_discovery --mass 1.2 --diagonal 0.45 --prop 0.254

Note on drag: stock x500 has NO body drag (<velocity_decay/> is empty), which measured
0.095-0.149 (m/s2)/(m/s) against a real Blackbird's 0.327-0.405. --drag adds linear
velocity_decay if you want to model an airframe that actually has some.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

GZ_MODELS = Path("/src/gs25122/PX4-Autopilot/Tools/simulation/gz/models")
AIRFRAMES = Path("/src/gs25122/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes")

# stock x500 reference, read from its model.sdf / airframe file
X500 = {"mass": 2.0, "rotor_mass": 0.016076923076923075, "motor_constant": 8.54858e-06,
        "moment_constant": 0.016, "max_rot_velocity": 1000.0,
        "rotor_drag": 8.06428e-05, "px": 0.13, "py": 0.22, "prop": 0.254,
        "thr_hover": 0.60}

ROTORS = [  # (index, x_sign, y_sign, direction, px4 KM sign)
    (0, +1, +1, "ccw", +1), (1, -1, -1, "ccw", +1),
    (2, +1, -1, "cw", -1), (3, -1, +1, "cw", -1),
]


ROTOR_THICKNESS = 0.005  # m



def stock_sensors() -> str:
    """Lift the imu / air_pressure / navsat blocks verbatim out of x500_base.

    A hand-written minimal <sensor type="imu"> without the full <imu> noise block does not
    produce data -- PX4 reports "Accel #0 fail: STALE!" and never gets an estimate. Copying
    the stock blocks also keeps the accelerometer noise identical to the airframe every
    measurement in this project was characterised against.
    """
    src = (GZ_MODELS / "x500_base" / "model.sdf").read_text()
    out = []
    for name in ("imu_sensor", "air_pressure_sensor", "navsat_sensor"):
        start = src.index(f'<sensor name="{name}"')
        end = src.index("</sensor>", start) + len("</sensor>")
        out.append("      " + src[start:end].strip())
    return "\n".join(out)


def rotor_link(i: int, x: float, y: float, z: float, rm: float, r_prop: float) -> str:
    # Solid cylinder, not an ideal thin disc. A thin disc gives Ixx + Iyy == Izz exactly,
    # and Gazebo rejects that: the inertia triangle inequality has to hold strictly
    # ("A link named rotor_0 has invalid inertia"). The thickness term breaks the tie.
    rad = r_prop / 2
    izz = 0.5 * rm * rad ** 2
    ixx = rm * (3 * rad ** 2 + ROTOR_THICKNESS ** 2) / 12
    return f"""
    <link name="rotor_{i}">
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
      <inertial>
        <mass>{rm:.6f}</mass>
        <inertia><ixx>{ixx:.3e}</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>{ixx:.3e}</iyy><iyz>0</iyz><izz>{izz:.3e}</izz></inertia>
      </inertial>
      <collision name="rotor_{i}_collision">
        <!-- A thin blade, matching x500. A full-disc collision at prop radius catches on
             spawn and tips the airframe onto its back before it is ever armed. -->
        <geometry><box><size>{r_prop:.4f} 0.017 0.0009</size></box></geometry>
      </collision>
      <visual name="rotor_{i}_visual">
        <geometry><cylinder><radius>{rad:.4f}</radius><length>{ROTOR_THICKNESS}</length></cylinder></geometry>
        <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
      </visual>
    </link>
    <joint name="rotor_{i}_joint" type="revolute">
      <parent>base_link</parent><child>rotor_{i}</child>
      <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit>
            <dynamics><damping>0.004</damping></dynamics></axis>
    </joint>"""


def motor_plugin(i: int, direction: str, kf: float, km: float, wmax: float,
                 drag: float) -> str:
    return f"""
  <plugin filename="gz-sim-multicopter-motor-model-system"
          name="gz::sim::systems::MulticopterMotorModel">
    <jointName>rotor_{i}_joint</jointName>
    <linkName>rotor_{i}</linkName>
    <turningDirection>{direction}</turningDirection>
    <timeConstantUp>0.0125</timeConstantUp>
    <timeConstantDown>0.025</timeConstantDown>
    <maxRotVelocity>{wmax}</maxRotVelocity>
    <motorConstant>{kf:.6e}</motorConstant>
    <momentConstant>{km}</momentConstant>
    <commandSubTopic>command/motor_speed</commandSubTopic>
    <motorNumber>{i}</motorNumber>
    <rotorDragCoefficient>{drag:.6e}</rotorDragCoefficient>
    <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
    <motorType>velocity</motorType>
  </plugin>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tbs_discovery")
    ap.add_argument("--mass", type=float, required=True, help="all-up mass [kg], battery in")
    ap.add_argument("--diagonal", type=float, required=True, help="motor-to-motor [m]")
    ap.add_argument("--prop", type=float, default=0.254, help="propeller diameter [m]")
    ap.add_argument("--body", type=float, nargs=3, default=None,
                    help="body box L W H [m]; defaults to diagonal-scaled")
    ap.add_argument("--drag", type=float, default=0.0,
                    help="linear velocity_decay; 0 = stock (no body drag, like x500)")
    ap.add_argument("--airframe-id", type=int, default=4100)
    args = ap.parse_args()

    r = args.diagonal / 2.0
    # keep the stock x500 arm aspect (it is a wide layout, not a symmetric 45-degree X)
    theta = math.atan2(X500["py"], X500["px"])
    px, py = r * math.cos(theta), r * math.sin(theta)

    rm = X500["rotor_mass"]
    body_mass = args.mass - 4 * rm
    L, W, H = args.body or (args.diagonal * 0.5, args.diagonal * 0.5, 0.08)

    # inertia: 4 point-mass motors at their real positions + a uniform box body
    ixx = 4 * rm * py ** 2 + body_mass * (W ** 2 + H ** 2) / 12
    iyy = 4 * rm * px ** 2 + body_mass * (L ** 2 + H ** 2) / 12
    izz = 4 * rm * (px ** 2 + py ** 2) + body_mass * (L ** 2 + W ** 2) / 12

    # hold the stock thrust-to-weight ratio so the vehicle flies at a sane hover throttle
    kf = X500["motor_constant"] * (args.mass / X500["mass"])
    hover_thr = X500["thr_hover"]

    decay = (f"<velocity_decay><linear>{args.drag}</linear><angular>0</angular></velocity_decay>"
             if args.drag > 0 else "<velocity_decay/>")

    sensors = stock_sensors()
    # Sit the body just clear of the ground rather than dropping it from a height.
    spawn_z = H / 2 + 0.02
    # Gazebo model frame is x-forward / y-LEFT / z-up; PX4's allocator is FRD, y-RIGHT.
    # The SDF y therefore carries the opposite sign to CA_ROTORn_PY -- stock x500 puts
    # rotor_0 at SDF y=-0.174 while its CA_ROTOR0_PY is +0.22. Using one sign for both
    # mirrors the airframe, inverts roll control, and the vehicle skids along the ground
    # under full thrust instead of climbing.
    links = "".join(rotor_link(i, sx * px, -sy * py, 0.03, rm, args.prop)
                    for i, sx, sy, _, _ in ROTORS)
    plugins = "".join(motor_plugin(i, d, kf, X500["moment_constant"],
                                   X500["max_rot_velocity"], X500["rotor_drag"])
                      for i, _, _, d, _ in ROTORS)

    sdf = f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version='1.9'>
  <model name='{args.name}'>
    <pose>0 0 {spawn_z:.3f} 0 0 0</pose>
    <self_collide>false</self_collide>
    <static>false</static>
    <link name="base_link">
      <inertial>
        <mass>{body_mass:.4f}</mass>
        <inertia><ixx>{ixx:.6f}</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>{iyy:.6f}</iyy><iyz>0</iyz><izz>{izz:.6f}</izz></inertia>
      </inertial>
      <gravity>true</gravity>
      {decay}
      <collision name="base_collision">
        <geometry><box><size>{L:.3f} {W:.3f} {H:.3f}</size></box></geometry>
      </collision>
      <visual name="base_visual">
        <geometry><box><size>{L:.3f} {W:.3f} {H:.3f}</size></box></geometry>
        <material><ambient>0.15 0.15 0.15 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material>
      </visual>
{sensors}
    </link>{links}
{plugins}
  </model>
</sdf>
"""

    mdir = GZ_MODELS / args.name
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model.sdf").write_text(sdf)
    (mdir / "model.config").write_text(
        f"""<?xml version="1.0"?>
<model>
  <name>{args.name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Generated by mario_sitl/scripts/make_airframe.py from mass={args.mass} kg,
  motor-to-motor diagonal={args.diagonal} m, prop={args.prop} m. Inertia derived from
  geometry; thrust constant scaled from x500 to hold its thrust-to-weight ratio. Values not
  derivable from those inputs are inherited from x500.</description>
</model>
""")

    af = AIRFRAMES / f"{args.airframe_id}_gz_{args.name}"
    af.write_text(f"""#!/bin/sh
#
# @name Gazebo {args.name}
#
# @type Quadrotor
#
# Generated by mario_sitl/scripts/make_airframe.py

. ${{R}}etc/init.d/rc.mc_defaults

PX4_SIMULATOR=${{PX4_SIMULATOR:=gz}}
PX4_GZ_WORLD=${{PX4_GZ_WORLD:=default}}
PX4_SIM_MODEL=${{PX4_SIM_MODEL:={args.name}}}

param set-default SIM_GZ_EN 1

param set-default SENS_EN_GPSSIM 1
param set-default SENS_EN_BAROSIM 0
param set-default SENS_EN_MAGSIM 1

param set-default CA_AIRFRAME 0
param set-default CA_ROTOR_COUNT 4

param set-default CA_ROTOR0_PX {px:.4f}
param set-default CA_ROTOR0_PY {py:.4f}
param set-default CA_ROTOR0_KM  0.05

param set-default CA_ROTOR1_PX {-px:.4f}
param set-default CA_ROTOR1_PY {-py:.4f}
param set-default CA_ROTOR1_KM  0.05

param set-default CA_ROTOR2_PX {px:.4f}
param set-default CA_ROTOR2_PY {-py:.4f}
param set-default CA_ROTOR2_KM -0.05

param set-default CA_ROTOR3_PX {-px:.4f}
param set-default CA_ROTOR3_PY {py:.4f}
param set-default CA_ROTOR3_KM -0.05

param set-default SIM_GZ_EC_FUNC1 101
param set-default SIM_GZ_EC_FUNC2 102
param set-default SIM_GZ_EC_FUNC3 103
param set-default SIM_GZ_EC_FUNC4 104

param set-default SIM_GZ_EC_MIN1 150
param set-default SIM_GZ_EC_MIN2 150
param set-default SIM_GZ_EC_MIN3 150
param set-default SIM_GZ_EC_MIN4 150

param set-default SIM_GZ_EC_MAX1 1000
param set-default SIM_GZ_EC_MAX2 1000
param set-default SIM_GZ_EC_MAX3 1000
param set-default SIM_GZ_EC_MAX4 1000

param set-default MPC_THR_HOVER {hover_thr}
""")

    cml = AIRFRAMES / "CMakeLists.txt"
    txt = cml.read_text()
    entry = f"{args.airframe_id}_gz_{args.name}"
    if entry not in txt:
        txt = txt.replace("\t4001_gz_x500\n", f"\t4001_gz_x500\n\t{entry}\n", 1)
        cml.write_text(txt)

    thrust_max = 4 * kf * X500["max_rot_velocity"] ** 2
    print(f"model   : {mdir}")
    print(f"airframe: {af}  (PX4_SIM_MODEL={args.name}, SYS_AUTOSTART={args.airframe_id})")
    print(f"\nDERIVED from your numbers:")
    print(f"  rotor positions   (+-{px:.4f}, +-{py:.4f}) m   radius {r:.4f} m")
    print(f"  body mass         {body_mass:.4f} kg  (total {args.mass} - 4 rotors)")
    print(f"  inertia           Ixx {ixx:.5f}  Iyy {iyy:.5f}  Izz {izz:.5f} kg m^2")
    print(f"  motorConstant     {kf:.4e}  -> max thrust {thrust_max:.1f} N "
          f"= {thrust_max/(args.mass*9.80665):.2f} : 1 thrust/weight")
    print(f"\nINHERITED from x500 (estimates -- correct if you have real values):")
    print(f"  rotor mass        {rm*1000:.1f} g each")
    print(f"  momentConstant    {X500['moment_constant']}")
    print(f"  maxRotVelocity    {X500['max_rot_velocity']} rad/s")
    print(f"  rotorDragCoeff    {X500['rotor_drag']:.4e}")
    print(f"  arm aspect angle  {math.degrees(theta):.1f} deg from +x")
    print(f"  MPC_THR_HOVER     {hover_thr}")
    print(f"  body box          {L:.3f} x {W:.3f} x {H:.3f} m")
    print(f"  body drag         {'linear velocity_decay=' + str(args.drag) if args.drag else 'NONE (stock x500 behaviour)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
