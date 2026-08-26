"""3D trajectory plots and animations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from mario.evaluate import rollout_trajectory


def _new_axes(gt: np.ndarray, pred: np.ndarray, z_min=None, z_max=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    points = np.vstack([gt, pred])
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(
        points[:, 2].min() if z_min is None else z_min,
        points[:, 2].max() if z_max is None else z_max,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    return fig, ax


def plot_trajectory(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    title: str = "",
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Path:
    """Save a static 3D overlay of the ground-truth and predicted trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = _new_axes(gt, pred, z_min, z_max)
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "k-", lw=2.0, label="Blackbird GT")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], "r-", lw=1.6, label="Causal-MARIO")
    ax.legend(loc="upper left", fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def animate_trajectory(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    title: str = "",
    n_frames: int = 120,
    fps: int = 20,
    rotate: bool = True,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Path:
    """Save a GIF where both trajectories are drawn progressively over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.animation import FuncAnimation, PillowWriter

    rcParams["axes.unicode_minus"] = False

    n = len(gt)
    frames = np.linspace(0, n - 1, min(n_frames, n)).astype(int)

    fig, ax = _new_axes(gt, pred, z_min, z_max)
    # faint full path for context
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="gray", lw=0.8, alpha=0.25)

    gt_line, = ax.plot([], [], [], "k-", lw=2.2, label="Blackbird GT")
    pred_line, = ax.plot([], [], [], "r-", lw=1.8, label="Causal-MARIO")
    gt_dot, = ax.plot([], [], [], "ko", ms=7)
    pred_dot, = ax.plot([], [], [], "ro", ms=6)
    ax.legend(loc="upper left", fontsize=10)

    def update(frame_idx: int):
        i = frames[frame_idx]
        gt_line.set_data(gt[: i + 1, 0], gt[: i + 1, 1])
        gt_line.set_3d_properties(gt[: i + 1, 2])
        pred_line.set_data(pred[: i + 1, 0], pred[: i + 1, 1])
        pred_line.set_3d_properties(pred[: i + 1, 2])
        gt_dot.set_data([gt[i, 0]], [gt[i, 1]])
        gt_dot.set_3d_properties([gt[i, 2]])
        pred_dot.set_data([pred[i, 0]], [pred[i, 1]])
        pred_dot.set_3d_properties([pred[i, 2]])
        ax.set_title(f"{title}  |  t = {i + 1}/{n}", fontsize=13, fontweight="bold")
        if rotate:
            ax.view_init(elev=25, azim=-60 + 120 * frame_idx / len(frames))
        return gt_line, pred_line, gt_dot, pred_dot

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return out_path


def predict_and_plot(
    net,
    data,
    device,
    cfg,
    out_path: str | Path,
    title: str = "",
    animate: bool = True,
    **kwargs,
) -> Optional[Path]:
    """Roll out one sequence and render it, returning ``None`` if it is too short."""
    rollout: Optional[Tuple[np.ndarray, np.ndarray]] = rollout_trajectory(
        net,
        data,
        device,
        window_size=cfg.window_size,
        label_start_index=cfg.label_start_index,
        label_stride=cfg.label_stride,
    )
    if rollout is None:
        return None
    gt, pred = rollout
    render = animate_trajectory if animate else plot_trajectory
    return render(gt, pred, out_path, title=title, **kwargs)
