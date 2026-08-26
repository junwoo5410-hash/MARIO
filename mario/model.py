"""Causal Mamba network that predicts body-frame displacement and its covariance."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class CausalMambaBlock(nn.Module):
    """Pre-norm Mamba block. The residual connection is added by the caller."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(x))


class CNNEncoder(nn.Module):
    """Strided 1D CNN that downsamples a (B, C, T) sequence before the Mamba stack."""

    def __init__(
        self,
        in_ch: int,
        c_list: Sequence[int] = (32, 64),
        k_list: Sequence[int] = (7, 7),
        s_list: Sequence[int] = (3, 3),
        p_list: Sequence[int] = (3, 3),
        dropout: float = 0.5,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_prev = in_ch
        for c, k, s, p in zip(c_list, k_list, s_list, p_list):
            layers += [
                nn.Conv1d(c_prev, c, k, stride=s, padding=p),
                nn.BatchNorm1d(c),
                nn.GELU(),
            ]
            c_prev = c
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalMambaDispNet(nn.Module):
    """Fuses IMU, orientation and motor thrust into a per-step displacement estimate.

    Each encoder downsamples the 1000-sample input window by 9x (two stride-3 convs),
    so the network emits one displacement per ``label_stride`` input samples.
    """

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
        self.motor_encoder = CNNEncoder(in_ch=3)

        # 3 encoders x 64 output channels -> d_model
        self.fcn = nn.Sequential(nn.Linear(192, d_model))
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
        motor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """acc/gyro/rot_so3/motor are (B, T, 3); returns (disp, cov) of shape (B, T', 3)."""
        imu = torch.cat([acc, gyro], dim=-1)
        x1 = self.imu_encoder(imu.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(rot_so3.transpose(-1, -2)).transpose(-1, -2)
        x3 = self.motor_encoder(motor.transpose(-1, -2)).transpose(-1, -2)

        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.gelu(self.bn(self.fcn(x).transpose(-1, -2)).transpose(-1, -2))

        for mamba in self.mamba_layers:
            x = x + mamba(x)

        # exp(... - 5) keeps the initial covariance small and strictly positive
        return self.disp_decoder(x), torch.exp(self.cov_decoder(x) - 5.0)


def build_model(cfg) -> CausalMambaDispNet:
    """Instantiate the network from a :class:`mario.config.ModelConfig`."""
    return CausalMambaDispNet(
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        num_layers=cfg.num_layers,
    )
