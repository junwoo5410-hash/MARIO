"""Small helpers shared by the training and evaluation entry points."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (including CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Turn ``"auto"`` into cuda when available, otherwise honour the explicit choice."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device.index or 0)})"
    return str(device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
