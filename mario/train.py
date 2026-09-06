"""Training loop for :class:`~mario.model.CausalMambaDispNet`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import torch
import torch.utils.data as Data

from mario.config import Config
from mario.dataset import collate_fn
from mario.losses import huber_loss, uncertainty_loss


def _forward_batch(net, batch, device):
    """Run one batch and return the length-aligned (prediction, covariance, target)."""
    acc = batch["acc"].to(device)
    gyro = batch["gyro"].to(device)
    gt_disp = batch["gt_disp"].to(device)
    rot_so3 = batch["gt_rot"].to(device).Log().tensor().float()

    pred_disp, pred_cov = net(acc, gyro, rot_so3)
    # the encoder stack can emit one extra step; trim both sides to the common length
    n = min(pred_disp.shape[1], gt_disp.shape[1])
    return pred_disp[:, :n], pred_cov[:, :n], gt_disp[:, :n]


@torch.no_grad()
def evaluate_rmse(net, loader, device) -> float:
    """Mean per-batch RMSE of the predicted displacement vectors."""
    net.eval()
    total, count = 0.0, 0
    for batch in loader:
        pred_disp, _, gt_disp = _forward_batch(net, batch, device)
        total += torch.sqrt(((pred_disp - gt_disp).norm(dim=-1) ** 2).mean()).item()
        count += 1
    return total / max(count, 1)


def train(
    net: torch.nn.Module,
    train_dataset: Data.Dataset,
    test_dataset: Data.Dataset,
    cfg: Config,
    device: torch.device,
) -> Dict[str, object]:
    """Train ``net``, checkpointing whenever the test RMSE improves.

    Returns a history dict and leaves the best weights at ``cfg.checkpoint_path``.
    """
    tcfg = cfg.train
    output_root = cfg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=tcfg.num_workers,
    )
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=tcfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=tcfg.num_workers,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=tcfg.scheduler_factor,
        patience=tcfg.scheduler_patience,
        min_lr=tcfg.scheduler_min_lr,
    )

    history: list[Dict[str, float]] = []
    best_rmse = float("inf")
    started = time.time()

    for epoch in range(1, tcfg.epochs + 1):
        net.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            pred_disp, pred_cov, gt_disp = _forward_batch(net, batch, device)
            residual = pred_disp - gt_disp

            # the covariance head is trained on a detached residual so it cannot
            # bleed gradients into the displacement head
            loss = tcfg.loss_weight * huber_loss(residual, delta=tcfg.huber_delta)
            loss = loss + tcfg.uncertainty_weight * uncertainty_loss(residual.detach(), pred_cov)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tcfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        test_rmse = evaluate_rmse(net, test_loader, device)
        scheduler.step(test_rmse)

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(net.state_dict(), cfg.checkpoint_path)

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "test_rmse": test_rmse,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if epoch == 1 or epoch % tcfg.log_every == 0 or epoch == tcfg.epochs:
            print(
                f"Epoch {epoch:3d}/{tcfg.epochs} | "
                f"loss {history[-1]['train_loss']:.5f} | "
                f"test RMSE {test_rmse:.5f} | best {best_rmse:.5f} | "
                f"{time.time() - started:.0f}s"
            )

    elapsed = time.time() - started
    print(f"\nTraining finished in {elapsed / 60:.1f} min — best test RMSE {best_rmse:.5f}")
    print(f"Checkpoint: {cfg.checkpoint_path}")

    return {"best_test_rmse": best_rmse, "elapsed_sec": elapsed, "history": history}


def load_checkpoint(net: torch.nn.Module, path: str | Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(Path(path).expanduser(), map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net
