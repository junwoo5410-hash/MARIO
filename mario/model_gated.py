"""Gated IMU/orientation fusion — one swapped layer against CausalMambaDispNet.

Motivation. At constant velocity the accelerometer says almost nothing about speed: it
reads gravity alone, which is why hover is unobservable (APPLICATION_VALIDATION.md 3.4).
Attitude is not equally blind there. A multirotor holding a cruise leans into its own
drag, and the lean angle grows with speed (tan(theta) ~ k*v/(m*g)), so the orientation
input still carries speed information in exactly the regime where the IMU loses it.

CausalMambaDispNet fuses the two encoders by concatenation into one Linear(128 -> 64):
the mixing weights are fixed once training ends and cannot depend on what the drone is
doing at that instant. This class replaces that layer with a gate that is recomputed at
every step:

    g = sigmoid(Linear(128 -> 64))      # per step, per channel
    x = g * x_imu + (1 - g) * x_ori

The rest of the network -- both encoders, the BatchNorm, the Mamba stack, both decoders
-- is untouched and imported from mario.model, so a difference between the two can only
come from the fusion. The parameter counts are identical (both layers are 128x64 + 64),
which is deliberate: this is not a capacity comparison.

Nothing in mario/model.py is modified (work rule 3); this is a separate class.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from mario.model import CausalMambaBlock, CNNEncoder


class GatedMambaDispNet(nn.Module):
    """CausalMambaDispNet with a per-step gate in place of the fusion Linear."""

    def __init__(
        self,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 1,
        num_layers: int = 1,
        d_model: int = 64,
    ):
        super().__init__()
        self.imu_encoder = CNNEncoder(in_ch=6)
        self.ori_encoder = CNNEncoder(in_ch=3)

        # same shape as CausalMambaDispNet.fcn, read as a mixing weight instead of a
        # projection: 1 keeps the IMU feature, 0 keeps the orientation feature
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.bn = nn.BatchNorm1d(d_model)
        self.gelu = nn.GELU()

        self.mamba_layers = nn.ModuleList(
            [CausalMambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)]
        )

        self.disp_decoder = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 3))
        self.cov_decoder = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(
        self,
        acc: torch.Tensor,
        gyro: torch.Tensor,
        rot_so3: torch.Tensor,
        return_gate: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """acc/gyro/rot_so3 are (B, T, 3).

        Returns ``(disp, cov)`` like CausalMambaDispNet, so the existing training loop and
        evaluators work unchanged; ``return_gate=True`` additionally returns the gate
        ``(B, T', d_model)`` for the diagnostics in eval_gated.py.
        """
        imu = torch.cat([acc, gyro], dim=-1)
        x1 = self.imu_encoder(imu.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(rot_so3.transpose(-1, -2)).transpose(-1, -2)

        g = self.gate(torch.cat([x1, x2], dim=-1))
        x = g * x1 + (1.0 - g) * x2
        x = self.gelu(self.bn(x.transpose(-1, -2)).transpose(-1, -2))

        for mamba in self.mamba_layers:
            x = x + mamba(x)

        disp = self.disp_decoder(x)
        cov = torch.exp(self.cov_decoder(x) - 5.0)
        return (disp, cov, g) if return_gate else (disp, cov)


def build_model(cfg) -> GatedMambaDispNet:
    """Instantiate from a :class:`mario.config.ModelConfig`, mirroring mario.model."""
    return GatedMambaDispNet(
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        num_layers=cfg.num_layers,
    )
