#!/usr/bin/env python
"""Bidirectional variant of the MARIO network, for isolating what AirIO's edge comes from.

AirIO beats MARIO on the unseen trajectories (ATE 1.294 vs 1.584 seed-averaged) and its
sequence model is two bidirectional GRUs, so time t's estimate sees up to 10 s of future.
MARIO's Mamba is a left-to-right recurrence. This variant makes the only change be the
direction: same CNN encoders, same widths, same decoders, same everything else -- one extra
Mamba stack run over the reversed sequence, concatenated before the decoders.

If the unseen gap closes, directionality explains it and the gap is the price of being
deployable. If it does not, the cause is elsewhere and bidirectionality is not worth
chasing. Either way this cannot be flown: it needs samples that have not happened yet.

Kept out of mario/ deliberately (work rule 3) -- it composes the published blocks rather
than editing them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mario.model import CausalMambaBlock, CNNEncoder  # noqa: E402


class BiMambaDispNet(nn.Module):
    """Same as CausalMambaDispNet but the Mamba stack is run in both directions."""

    def __init__(self, d_state: int = 32, d_conv: int = 4, expand: int = 1,
                 num_layers: int = 1, d_model: int = 64):
        super().__init__()
        self.imu_encoder = CNNEncoder(in_ch=6)
        self.ori_encoder = CNNEncoder(in_ch=3)

        self.fcn = nn.Sequential(nn.Linear(128, d_model))
        self.bn = nn.BatchNorm1d(d_model)
        self.gelu = nn.GELU()

        self.fwd_layers = nn.ModuleList(
            [CausalMambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)])
        self.bwd_layers = nn.ModuleList(
            [CausalMambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)])

        # 2 * d_model: forward and reversed streams are concatenated, mirroring how a
        # bidirectional GRU doubles its hidden width before the head.
        self.disp_decoder = nn.Sequential(nn.Linear(2 * d_model, 128), nn.GELU(), nn.Linear(128, 3))
        self.cov_decoder = nn.Sequential(nn.Linear(2 * d_model, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor, rot_so3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
        imu = torch.cat([acc, gyro], dim=-1)
        x1 = self.imu_encoder(imu.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(rot_so3.transpose(-1, -2)).transpose(-1, -2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.gelu(self.bn(self.fcn(x).transpose(-1, -2)).transpose(-1, -2))

        f = x
        for m in self.fwd_layers:
            f = f + m(f)
        b = torch.flip(x, dims=[1])
        for m in self.bwd_layers:
            b = b + m(b)
        b = torch.flip(b, dims=[1])

        h = torch.cat([f, b], dim=-1)
        return self.disp_decoder(h), torch.exp(self.cov_decoder(h) - 5.0)
