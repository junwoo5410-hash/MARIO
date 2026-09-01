#!/usr/bin/env python
"""Stage 6 — sweep the closed loop until it breaks, and record where.

One factor at a time rather than the full 4x3x4 grid: 48 closed-loop flights is hours of
wall clock, and the axes are not expected to interact in any way the marginals would hide.
Each run restarts PX4 and Gazebo from scratch, because a run that ends displaced (or on its
side) contaminates the next one.

Every run is closed loop on MARIO alone -- GPS off, velocity-only external vision. The GPS
baseline (Stage 1: 0.126 m arrival, 0.59 m settled deviation) is the yardstick; these
numbers are meaningless in isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sweep"
PX4_ROOTFS = Path("/src/gs25122/PX4-Autopilot/build/px4_sitl_default/rootfs")
PX4_BIN = Path("/src/gs25122/PX4-Autopilot/build/px4_sitl_default/bin/px4")
PX4_ENV = Path("/src/gs25122/miniconda3/envs/px4")

# distance [m], hover [s], cruise speed [m/s]; 0 speed = stepped setpoint
GRID = (
    [{"north": d, "hover": 10, "speed": 0.0, "axis": "distance"} for d in (2, 5, 10, 20)]
    + [{"north": 5, "hover": h, "speed": 0.0, "axis": "hover"} for h in (30, 60, 120)]
    + [{"north": 10, "hover": 10, "speed": v, "axis": "speed"} for v in (0.5, 1.0, 2.0)]
)


def sh(cmd: str, timeout: float = 120) -> None:
    """Run a shell command with output to /dev/null.

    NOT capture_output=True: a backgrounded PX4 inherits the captured stdout pipe, so
    communicate() blocks on a pipe that never closes and the timeout fires even though the
    process was launched with nohup and '&'. That killed the first sweep on case 1.
    """
    with open(os.devnull, "wb") as null:
        subprocess.run(["bash", "-lc", cmd], stdout=null, stderr=null,
                       timeout=timeout, check=False)


def kill_sim() -> None:
    # kill by pid: a pkill pattern can match this script's own shell and take it down.
    for pat in ("bin/px4 -d", "gz sim"):
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout
        for pid in out.split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        time.sleep(2)


def restart_px4(log: Path, boot_timeout: float = 60.0) -> bool:
    kill_sim()
    if log.exists():
        log.unlink()
    sh(f"cd {PX4_ROOTFS} && nohup env PATH={PX4_ENV}/bin:$PATH "
       f"LD_LIBRARY_PATH={PX4_ENV}/lib HEADLESS=1 PX4_GZ_MODEL_POSE='0,0,0,0,0,0' "
       f"PX4_SIM_MODEL=gz_x500 {PX4_BIN} -d > {log} 2>&1 < /dev/null &", timeout=30)

    # Poll for readiness rather than sleeping a fixed amount: a fixed sleep either wastes
    # time or misses a slow boot, and both were happening.
    deadline = time.time() + boot_timeout
    while time.time() < deadline:
        time.sleep(2)
        if log.exists() and "Ready for takeoff" in log.read_text(errors="ignore"):
            return True
    return log.exists() and "data writer" in log.read_text(errors="ignore")


def run_case(case: dict, idx: int, ckpt: Path) -> dict | None:
    tag = f"{case['axis']}_d{case['north']}_h{case['hover']}_v{case['speed']}"
    out = RESULTS / f"{tag}.json"
    px4_log = RESULTS / f"{tag}_px4.log"
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"\n[{idx}] {tag}", flush=True)
    if not restart_px4(px4_log):
        print("   PX4 failed to start", flush=True)
        return None

    duration = 60 + case["hover"] + case["north"] * 2
    script = f"""
source /src/gs25122/miniconda3/etc/profile.d/conda.sh
conda activate ros2
source /src/gs25122/px4_ros2_ws/install/setup.bash
cd {ROOT}
python mario_ros/mario_odometry_node.py --rate 25 --duration {duration + 40} --publish \
    --out {RESULTS / (tag + '_shadow.json')} &
M=$!
sleep 25
python mario_ros/offboard_mission_node.py --out {out} --north {case['north']} \
    --hover {case['hover']} --cruise-speed {case['speed']}
kill $M 2>/dev/null; wait $M 2>/dev/null
"""
    (RESULTS / f"{tag}_run.sh").write_text(script)
    sh(f"bash {RESULTS / (tag + '_run.sh')} > {RESULTS / (tag + '_run.log')} 2>&1",
       timeout=duration + 180)

    if not out.exists():
        print("   no mission log produced", flush=True)
        return None

    d = json.loads(out.read_text())
    s = d["summary"]
    events = [e["phase"] for e in d.get("events", [])]
    armed = "Armed by external command" in px4_log.read_text(errors="ignore")
    crashed = "Disarmed by failsafe" in px4_log.read_text(errors="ignore")
    row = {**case, "tag": tag, "armed": armed, "crashed": crashed, "phases": events,
           "gt_arrival_error": s.get("gt_arrival_error"),
           "gt_final_error": s.get("gt_final_error"),
           "gt_hover_drift": s.get("gt_hover_drift"),
           "ekf_tracking_error": s.get("ekf_tracking_error"),
           "passed": s.get("passed")}
    print(f"   armed={armed} crashed={crashed} arrival="
          f"{row['gt_arrival_error']} drift={row['gt_hover_drift']}", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/src/gs25122/MARIO/runs/sitl_agg/best.pt"))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "stage6_sweep.json")
    ap.add_argument("--only", default=None, help="restrict to one axis")
    args = ap.parse_args()

    grid = [c for c in GRID if args.only is None or c["axis"] == args.only]
    rows = []
    t0 = time.time()
    for i, case in enumerate(grid, 1):
        # One bad case must not cost the other nine.
        try:
            r = run_case(case, i, args.ckpt)
        except Exception as exc:
            print(f"   case failed: {type(exc).__name__}: {exc}", flush=True)
            r = None
        if r:
            rows.append(r)
            args.out.write_text(json.dumps(rows, indent=2))   # checkpoint as we go
    print(f"\n{len(rows)}/{len(grid)} runs in {(time.time()-t0)/60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
