#!/usr/bin/env python
"""The floor every trajectory estimator has to clear: predict zero displacement.

A model that outputs nothing at all leaves the estimate parked at the anchor, so its
ATE is the RMS distance the drone travelled away from its starting pose. A result above
this number is worse than saying "it never moved", which makes it the only honest
reference point for a transfer result.

The pose grid has to be the one the real evaluation produces, and that grid is subtle
(see eval_full_rollout), so this reuses ``rollout_full`` itself and only forces the
predicted displacement to zero. Nothing about the chaining is re-derived here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config, DEFAULT_SEEN, DEFAULT_UNSEEN   # noqa: E402
from mario.data import load_split as load_blackbird             # noqa: E402
from mario.model import CausalMambaDispNet                      # noqa: E402
from eval_full_rollout import rollout_full, metrics             # noqa: E402
import euroc_data                                               # noqa: E402


class ZeroDisp(torch.nn.Module):
    """Same output shape as the real network, all displacements zero."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, *args, **kwargs):
        disp, cov = self.net(*args, **kwargs)
        return torch.zeros_like(disp), cov


def score(pairs, net, dev, cfg) -> dict:
    per = {}
    for name, seq in pairs:
        r = rollout_full(net, seq, dev, cfg.data.window_size,
                         cfg.data.label_start_index, cfg.data.label_stride)
        if r is not None:
            per[name] = metrics(*r)
    return {"per_trajectory": per,
            "mean_ate": float(np.mean([m["ATE"] for m in per.values()])),
            "mean_tde": float(np.mean([m["TDE"] for m in per.values()]))}


def main() -> int:
    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ZeroDisp(CausalMambaDispNet(
        d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
        expand=cfg.model.expand, num_layers=cfg.model.num_layers)).to(dev).eval()

    out: Dict[str, dict] = {}
    for split, trajs in (("blackbird_seen", DEFAULT_SEEN),
                         ("blackbird_unseen", DEFAULT_UNSEEN)):
        pairs = [load_blackbird("/src/gs25122/blackbird_data", [t], "eval",
                                dt=cfg.data.dt, verbose=False)[0] for t in trajs]
        out[split] = score(pairs, net, dev, cfg)

    test = euroc_data.read_lists()["test"]
    out["euroc_test"] = score(
        euroc_data.load_split(test, dt=cfg.data.dt, verbose=False), net, dev, cfg)

    dest = ROOT / "mario_sitl" / "results" / "stationary_baseline.json"
    dest.write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        d = np.mean([m["distance"] for m in v["per_trajectory"].values()])
        n = np.mean([m["n_poses"] for m in v["per_trajectory"].values()])
        print(f"{k:18} ATE {v['mean_ate']:7.3f}   TDE {v['mean_tde']:7.2f} %   "
              f"mean {d:6.1f} m / {n:5.0f} poses   ({len(v['per_trajectory'])} traj)")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
