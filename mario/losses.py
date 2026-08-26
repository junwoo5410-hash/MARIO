"""Training objectives for displacement regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def huber_loss(residual: torch.Tensor, delta: float = 0.002) -> torch.Tensor:
    """Huber loss pulling the displacement residual towards zero."""
    return F.huber_loss(residual, torch.zeros_like(residual), delta=delta)


def uncertainty_loss(residual: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood used to calibrate the predicted covariance.

    Call with a detached ``residual`` so this term only trains the covariance head.
    """
    return ((residual.pow(2) / cov) + torch.log(cov)).mean()
