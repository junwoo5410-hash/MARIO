#!/usr/bin/env python
"""Stage 0 addendum — what to feed the `motor` input in SITL.

MARIO_closedloop_prompt.md 함정 A prescribes pinning `motor` to zeros for the SITL work.
That path exists in data.py, but it was never exercised during training: every Blackbird
sequence ships thrust_data.csv, so the network only ever saw

    motor = (0, 0, ~-0.82)

Feeding zeros is therefore out of distribution, not a neutral ablation. This script
measures the cost on the Blackbird eval split and scores two deployable substitutes.

What the three thrust columns actually are (verified against the IMU, not the header):
  col1, col2  identically zero in every train/eval sequence
  col3        body-z mass-normalised collective thrust [m/s^2],
              corr(col3, acc_z) = +0.92, mean diff 0.16 m/s^2
The file header claims "thrust_1..thrust_4"; the rows carry three values. Likewise
imu_data.csv's header lists acc before gyro while the data is gyro first — data.py
reads it correctly.

Read-only with respect to MARIO/mario/ (work rule 3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mario.config import Config  # noqa: E402
from mario.data import load_eval_sequences  # noqa: E402
from mario.evaluate import evaluate_trajectory  # noqa: E402
from mario.model import build_model  # noqa: E402

RUN_DIR = REPO_ROOT / "runs" / "trial8_100ep"

#: training-set mean of the normalised body-z thrust channel
TRAIN_MEAN_CH2 = -0.82
#: fixed divisor for the causal acc_z variant; ~max |specific thrust| seen in training
ACCZ_SCALE = 13.0

VARIANTS = {
    "orig": "thrust_data.csv as data.py normalises it (non-causal, needs the topic)",
    "zeros": "함정 A as written — motor pinned to 0",
    "const": f"ch2 pinned to the training mean {TRAIN_MEAN_CH2}",
    "accz": f"ch2 = clip(acc_z / {ACCZ_SCALE}, -1, 1) — causal, IMU-only",
}


def build_motor(data: dict, kind: str) -> torch.Tensor:
    if kind == "orig":
        return data["motor"]
    motor = torch.zeros_like(data["motor"])
    if kind == "const":
        motor[:, 2] = TRAIN_MEAN_CH2
    elif kind == "accz":
        motor[:, 2] = (data["acc"][:, 2] / ACCZ_SCALE).clamp(-1.0, 1.0)
    return motor


def main() -> int:
    cfg = Config.load(RUN_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = build_model(cfg.model).to(device).float()
    net.load_state_dict(torch.load(RUN_DIR / "best.pt", map_location=device, weights_only=True))
    net.eval()

    seen, unseen = load_eval_sequences(cfg.data, verbose=False)
    sequences = [("seen", n, d) for n, d in seen] + [("unseen", n, d) for n, d in unseen]

    kinds = list(VARIANTS)
    print(f"{'trajectory':<14}{'split':<8}" + "".join(f"{k:>10}" for k in kinds) + "   [ATE m]")
    print("-" * (22 + 10 * len(kinds)))

    results: dict = {"variants": VARIANTS, "per_trajectory": {}}
    per_kind: dict[str, list[float]] = {k: [] for k in kinds}

    for split, name, data in sequences:
        row = {}
        for kind in kinds:
            probe = dict(data)
            probe["motor"] = build_motor(data, kind)
            metrics = evaluate_trajectory(net, probe, device, cfg.data)
            ate = float("nan") if metrics is None else metrics["ATE"]
            row[kind] = ate
            per_kind[kind].append(ate)
        results["per_trajectory"][name] = {"split": split, **row}
        print(f"{name:<14}{split:<8}" + "".join(f"{row[k]:>10.3f}" for k in kinds))

    print("-" * (22 + 10 * len(kinds)))
    means = {k: float(np.nanmean(v)) for k, v in per_kind.items()}
    results["mean_ate"] = means
    print(f"{'mean':<22}" + "".join(f"{means[k]:>10.3f}" for k in kinds))

    baseline = means["orig"]
    print("\n원본 대비:")
    for kind in kinds[1:]:
        delta = (means[kind] - baseline) / baseline * 100.0
        print(f"  {kind:<7} {means[kind]:.3f} m  ({delta:+.1f} %)  — {VARIANTS[kind]}")

    out = REPO_ROOT / "mario_sitl" / "results" / "motor_input_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
