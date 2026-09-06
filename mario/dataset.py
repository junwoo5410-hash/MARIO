"""Sliding-window dataset with body-frame displacement labels."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.utils.data as Data

from mario.data import Sequence


def body_frame_displacements(
    positions: torch.Tensor,
    rotations,
    n_labels: int,
    start_index: int,
    stride: int,
) -> torch.Tensor | None:
    """Displacement over each ``stride``-sample step, rotated into the body frame at ``t``.

    Returns ``None`` when the window is too short to produce a single label.
    """
    disps = []
    for i in range(n_labels):
        idx_t = start_index + i * stride
        idx_next = idx_t + stride
        if idx_next >= len(positions):
            break
        world_disp = positions[idx_next] - positions[idx_t]
        disps.append(rotations[idx_t].Inv() @ world_disp)

    if not disps:
        return None
    # pad the tail by repeating the last label so every window has the same length
    disps.extend([disps[-1]] * (n_labels - len(disps)))
    return torch.stack(disps)


class BlackbirdDispDataset(Data.Dataset):
    """Windows of raw sensor data paired with per-step body-frame displacements.

    Each item holds ``window_size`` input samples and ``n_labels`` displacement
    targets, one per ``label_stride`` input samples (matching the encoder stride).
    """

    def __init__(
        self,
        sequences: Iterable[Sequence],
        window_size: int = 1000,
        step_size: int = 3,
        label_start_index: int = 14,
        label_stride: int = 9,
    ):
        self.window_size = window_size
        self.n_labels = (window_size - 2) // label_stride + 1
        self.windows: List[Dict[str, torch.Tensor]] = []

        for data in sequences:
            seq_len = len(data["acc"])
            pos, rot = data["gt_translation"], data["gt_orientation"]

            for j in range(0, seq_len - window_size - step_size, step_size):
                # labels need one extra stride of lookahead beyond the window
                end = j + window_size + label_stride
                labels = body_frame_displacements(
                    pos[j:end], rot[j:end], self.n_labels, label_start_index, label_stride
                )
                if labels is None:
                    continue
                self.windows.append(
                    {
                        "acc": data["acc"][j : j + window_size],
                        "gyro": data["gyro"][j : j + window_size],
                        "gt_rot": rot[j : j + window_size],
                        "gt_disp": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.windows[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of windows, preserving pypose ``SO3`` types."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_datasets(
    train_sequences: Iterable[Tuple[str, Sequence]],
    test_sequences: Iterable[Tuple[str, Sequence]],
    cfg,
) -> Tuple[BlackbirdDispDataset, BlackbirdDispDataset]:
    """Build the train/test window datasets from a :class:`mario.config.DataConfig`."""
    common = dict(
        window_size=cfg.window_size,
        label_start_index=cfg.label_start_index,
        label_stride=cfg.label_stride,
    )
    train_ds = BlackbirdDispDataset(
        [d for _, d in train_sequences], step_size=cfg.train_step_size, **common
    )
    test_ds = BlackbirdDispDataset(
        [d for _, d in test_sequences], step_size=cfg.test_step_size, **common
    )
    return train_ds, test_ds
