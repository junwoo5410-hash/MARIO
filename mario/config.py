"""Typed configuration objects, loadable from YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Trajectories the model is trained on.
DEFAULT_SEEN: List[str] = [
    "clover/yawForward/maxSpeed5p0",
    "halfMoon/yawForward/maxSpeed4p0",
    "star/yawForward/maxSpeed5p0",
    "egg/yawForward/maxSpeed8p0",
    "winter/yawForward/maxSpeed4p0",
]

# Held-out trajectories, used for generalisation numbers only.
DEFAULT_UNSEEN: List[str] = [
    "ampersand/yawForward/maxSpeed2p0",
    "sid/yawForward/maxSpeed5p0",
    "oval/yawForward/maxSpeed4p0",
    "sphinx/yawForward/maxSpeed4p0",
    "bentDice/yawForward/maxSpeed3p0",
]


@dataclass
class ModelConfig:
    d_state: int = 32
    d_conv: int = 4
    expand: int = 1
    num_layers: int = 1


@dataclass
class DataConfig:
    data_dir: str = "~/blackbird_data"
    window_size: int = 1000
    train_step_size: int = 3
    test_step_size: int = 10
    #: index of the first labelled sample inside a window (encoder receptive-field offset)
    label_start_index: int = 14
    #: input samples between consecutive displacement labels (= total encoder stride)
    label_stride: int = 9
    #: IMU resampling period in seconds
    dt: float = 0.01
    seen: List[str] = field(default_factory=lambda: list(DEFAULT_SEEN))
    unseen: List[str] = field(default_factory=lambda: list(DEFAULT_UNSEEN))


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.00025757267464272164
    weight_decay: float = 1.0e-3
    loss_weight: float = 1000.0
    huber_delta: float = 0.002
    uncertainty_weight: float = 1.0e-4
    grad_clip: float = 1.0
    scheduler_factor: float = 0.2
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1.0e-5
    num_workers: int = 0
    log_every: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "~/causal_mamba_disp_trial8_results"
    device: str = "auto"
    seed: int = 42

    # -- paths ------------------------------------------------------------
    @property
    def data_root(self) -> Path:
        return Path(self.data.data_dir).expanduser()

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser()

    @property
    def checkpoint_path(self) -> Path:
        return self.output_root / "best.pt"

    @property
    def results_path(self) -> Path:
        return self.output_root / "results.json"

    # -- (de)serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "Config":
        raw = dict(raw or {})
        sections = {
            "model": ModelConfig,
            "data": DataConfig,
            "train": TrainConfig,
        }
        kwargs: Dict[str, Any] = {}
        for name, section_cls in sections.items():
            kwargs[name] = section_cls(**(raw.pop(name, None) or {}))
        return cls(**kwargs, **raw)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load a YAML config; missing keys fall back to the Trial 8 defaults."""
        if path is None:
            return cls()
        with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
