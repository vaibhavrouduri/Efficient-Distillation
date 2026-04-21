"""Checkpoint utilities."""

from __future__ import annotations

import os
import shutil
from typing import Tuple

import torch
from utils.distributed import unwrap_model


def save_checkpoint(
    path: str,
    epoch: int,
    student,
    adapter,
    optimiser,
    scheduler,
    best_metric: float,
    is_best: bool = False,
) -> None:
    """
    Save training state to ``path``.  If ``is_best=True``, also copies the
    file to ``<dir>/best.pth``.
    """
    state = {
        "epoch":        epoch,
        "student":      unwrap_model(student).state_dict(),
        "adapter":      unwrap_model(adapter).state_dict(),
        "optimiser":    optimiser.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "best_metric":  best_metric,
    }
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best.pth")
        shutil.copyfile(path, best_path)


def load_checkpoint(
    path: str,
    student,
    adapter,
    optimiser=None,
    scheduler=None,
) -> Tuple[int, float]:
    """
    Load training state from ``path``.

    Returns:
        (start_epoch, best_metric)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    unwrap_model(student).load_state_dict(ckpt["student"])
    unwrap_model(adapter).load_state_dict(ckpt["adapter"])
    if optimiser is not None and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", 0.0)
