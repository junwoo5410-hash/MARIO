#!/usr/bin/env python
"""Stage 0 — environment gate.

Verifies every prerequisite listed in section 8 of MARIO_closedloop_prompt.md before
any SITL work starts. Exits non-zero if a hard gate fails (no GPU, no mamba-ssm, no
checkpoint), because the later stages cannot produce meaningful numbers without them.

Read-only with respect to MARIO/mario/ (work rule 3).
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Fixed constants from section 2 of the spec — the node must match these exactly.
WINDOW_SIZE = 1000
DT = 0.01
LABEL_START_INDEX = 14
LABEL_STRIDE = 9
N_LABELS = 111
CHECKPOINT = REPO_ROOT / "runs" / "trial8_100ep" / "best.pt"

OK, FAIL, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"
report: dict = {}
hard_failures: list[str] = []


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


def check_torch() -> None:
    section("1. torch / CUDA")
    import torch

    cuda = torch.cuda.is_available()
    report["torch"] = {"version": torch.__version__, "cuda_available": cuda}
    print(f"{OK} torch {torch.__version__}")
    if not cuda:
        # Spec: "GPU가 없으면 여기서 중단하고 보고할 것."
        print(f"{FAIL} CUDA unavailable — mamba-ssm needs a CUDA toolchain. Stop here.")
        hard_failures.append("no CUDA")
        return
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    report["torch"]["devices"] = names
    print(f"{OK} CUDA available — {len(names)} device(s): {names[0]}")


def check_mamba() -> None:
    section("2. mamba-ssm")
    try:
        mamba = importlib.import_module("mamba_ssm")
    except Exception as exc:  # noqa: BLE001 - report whatever import raised
        print(f"{FAIL} import mamba_ssm: {type(exc).__name__}: {exc}")
        report["mamba_ssm"] = {"importable": False, "error": str(exc)}
        hard_failures.append("mamba-ssm not importable")
        return
    version = getattr(mamba, "__version__", "unknown")
    report["mamba_ssm"] = {"importable": True, "version": version}
    print(f"{OK} mamba_ssm {version}")

    import torch
    from mamba_ssm import Mamba

    blk = Mamba(d_model=64, d_state=32, d_conv=4, expand=1).cuda()
    x = torch.randn(2, N_LABELS + 1, 64, device="cuda")
    y = blk(x)
    y.sum().backward()
    print(f"{OK} real CUDA kernel fwd+bwd {tuple(x.shape)} -> {tuple(y.shape)}")


def check_checkpoint() -> None:
    section("3. best.pt dummy inference")
    if not CHECKPOINT.exists():
        print(f"{FAIL} checkpoint missing: {CHECKPOINT}")
        report["checkpoint"] = {"path": str(CHECKPOINT), "exists": False}
        hard_failures.append("no checkpoint")
        return

    import torch

    from mario.config import Config
    from mario.model import build_model

    cfg_path = CHECKPOINT.parent / "config.yaml"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config()
    net = build_model(cfg.model).cuda().float()
    state = torch.load(CHECKPOINT, map_location="cuda", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    dummy = [torch.zeros(1, WINDOW_SIZE, 3, device="cuda") for _ in range(4)]
    with torch.no_grad():
        disp, cov = net(*dummy)

    n_params = sum(p.numel() for p in net.parameters())
    report["checkpoint"] = {
        "path": str(CHECKPOINT),
        "exists": True,
        "params": n_params,
        "disp_shape": list(disp.shape),
        "cov_shape": list(cov.shape),
        "spec_n_labels": N_LABELS,
    }
    print(f"{OK} loaded {CHECKPOINT.relative_to(REPO_ROOT)} ({n_params:,} params)")
    print(f"{OK} 4 x (1,{WINDOW_SIZE},3) -> disp {tuple(disp.shape)}, cov {tuple(cov.shape)}")

    # The spec states (1,111,3). The encoder actually emits 112 steps for a 1000-sample
    # window; mario/train.py trims to the label count. The ROS node must do the same.
    if disp.shape[1] != N_LABELS:
        print(
            f"{WARN} model emits {disp.shape[1]} steps, spec/labels use {N_LABELS}."
            f" The node must take the first {N_LABELS} (mario/train.py:27 does this)."
        )
        report["checkpoint"]["step_mismatch"] = True

    if not torch.isfinite(disp).all() or not (cov > 0).all():
        print(f"{FAIL} non-finite disp or non-positive cov")
        hard_failures.append("bad inference output")
    else:
        print(f"{OK} disp finite, cov strictly positive (cov range "
              f"{cov.min().item():.3e} .. {cov.max().item():.3e})")


def check_sitl_stack() -> None:
    section("4. SITL stack (ROS 2 / PX4 / Gazebo)")
    found = {}

    ros_distro = None
    for path in sorted(Path("/opt/ros").glob("*")) if Path("/opt/ros").exists() else []:
        ros_distro = path.name
    if shutil.which("ros2") or ros_distro:
        print(f"{OK} ROS 2 ({ros_distro or 'on PATH'})")
        found["ros2"] = ros_distro or "on PATH"
    else:
        print(f"{WARN} ROS 2 not found (needed from Stage 1)")
        found["ros2"] = None

    px4 = next((p for p in [REPO_ROOT.parent / "PX4-Autopilot", Path.home() / "PX4-Autopilot"]
                if p.exists()), None)
    if px4:
        try:
            ver = subprocess.run(["git", "-C", str(px4), "describe", "--tags"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            ver = "unknown"
        print(f"{OK} PX4-Autopilot at {px4} ({ver})")
        found["px4"] = {"path": str(px4), "version": ver}
    else:
        print(f"{WARN} PX4-Autopilot not found (needed from Stage 1)")
        found["px4"] = None

    for tool in ("gz", "MicroXRCEAgent"):
        where = shutil.which(tool)
        print(f"{OK} {tool}: {where}" if where else f"{WARN} {tool} not found")
        found[tool] = where

    report["sitl_stack"] = found


def main() -> int:
    print(f"MARIO Stage 0 — environment gate\npython: {sys.executable}")
    check_torch()
    if "no CUDA" not in hard_failures:
        check_mamba()
    if not hard_failures:
        check_checkpoint()
    check_sitl_stack()

    section("결과")
    out = REPO_ROOT / "mario_sitl" / "results" / "check_env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report["hard_failures"] = hard_failures
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if hard_failures:
        print(f"{FAIL} 하드 게이트 실패: {', '.join(hard_failures)}")
        print("  -> 해결 전까지 Stage 1 이후로 진행 금지 (작업 규칙 1).")
    else:
        print(f"{OK} MARIO 추론 게이트 통과 — Stage 1 진행 가능 조건 충족")
    missing = [k for k, v in report.get("sitl_stack", {}).items() if not v]
    if missing:
        print(f"{WARN} SITL 스택 미설치: {', '.join(missing)} — Stage 1 착수 전 구축 필요")
    print(f"\n  report: {out}")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
