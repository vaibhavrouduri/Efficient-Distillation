"""
DepthTrainer: trains the DPT depth head on top of the frozen distilled student.

Only the DPT head (neck + prediction layers) is optimised.
The student backbone stays frozen and in eval mode throughout.

Loss:
    Scale-Invariant Log loss (SILog) — the standard for monocular depth.
    L = sqrt(Var(d) + lambda * Mean(d)^2)   where d = log(pred) - log(gt)
    lambda=0.85 follows AdaBins / DPT convention.

Metrics (reported on validation):
    RMSE     root mean squared error (metres)
    AbsRel   mean absolute relative error
    delta1   fraction of pixels with max(pred/gt, gt/pred) < 1.25
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from utils.logger import setup_logger
from utils.distributed import (
    is_main_process,
    reduce_mean,
    reduce_tensor_sum,
    unwrap_model,
    wrap_ddp,
)


class DepthTrainer:
    """
    Trains the DPT depth head (models.depth_model.StudentWithDPT).

    Args:
        model:        StudentWithDPT instance (student frozen, DPT head trainable).
        train_loader: DataLoader yielding {"images": ..., "depths": ...}.
        val_loader:   same format, no augmentation.
        cfg:          config dict (from depth_config.yaml).
        device:       torch.device.
        wandb_run:    optional W&B run object.
    """

    SILOG_LAMBDA = 0.85

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        device:       torch.device,
        wandb_run=None,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.wandb_run    = wandb_run
        self.logger       = setup_logger("depth_trainer")
        self.is_main_process = is_main_process()
        self.distributed = cfg.get("distributed", False)

        if self.distributed:
            self.model = wrap_ddp(self.model, device)

        # Only optimise DPT head parameters
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

        self.optimiser = AdamW(
            trainable,
            lr           = cfg.get("lr", 1e-4),
            weight_decay = cfg.get("weight_decay", 1e-2),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max   = cfg.get("epochs", 50),
            eta_min = cfg.get("lr_min", 1e-6),
        )
        self.scaler = GradScaler(enabled=cfg.get("amp", True))

        self.epochs      = cfg.get("epochs", 50)
        self.log_every   = cfg.get("log_every", 20)
        self.save_dir    = Path(cfg.get("save_dir", "checkpoints/depth"))
        self.grad_clip   = cfg.get("grad_clip", 1.0)
        self.amp         = cfg.get("amp", True)

        self.start_epoch  = 0
        self.best_rmse    = float("inf")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        if cfg.get("resume"):
            self._resume(cfg["resume"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs):
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch)
            val_metrics   = self._validate(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"rmse={val_metrics['rmse']:.4f}  "
                f"abs_rel={val_metrics['abs_rel']:.4f}  "
                f"delta1={val_metrics['delta1']:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"time={elapsed:.1f}s"
            )

            if self.wandb_run is not None and self.is_main_process:
                self.wandb_run.log({
                    "epoch":          epoch + 1,
                    "train/loss":     train_metrics["loss"],
                    "val/rmse":       val_metrics["rmse"],
                    "val/abs_rel":    val_metrics["abs_rel"],
                    "val/delta1":     val_metrics["delta1"],
                    "lr":             self.scheduler.get_last_lr()[0],
                })

            # Save latest checkpoint
            if self.is_main_process:
                self._save(epoch, val_metrics["rmse"])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        # Ensure student backbone stays frozen+eval
        if hasattr(self.model, "student"):
            self.model.student.eval()

        total_loss = 0.0
        n_batches  = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device, non_blocking=True)
            depths = batch["depths"].to(self.device, non_blocking=True)

            self.optimiser.zero_grad()

            with autocast(enabled=self.amp):
                pred = self.model(images)          # (B, 1, H, W)
                pred = pred.squeeze(1)             # (B, H, W)
                loss = self._silog_loss(pred, depths)

            self.scaler.scale(loss).backward()

            if self.grad_clip:
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )

            self.scaler.step(self.optimiser)
            self.scaler.update()

            total_loss += loss.item()

            if (i + 1) % self.log_every == 0:
                self.logger.info(
                    f"  [{epoch+1}][{i+1}/{n_batches}]  loss={loss.item():.4f}"
                )

        avg_loss = total_loss / n_batches
        avg_loss = reduce_mean(avg_loss, self.device)
        return {"loss": avg_loss}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()

        rmse_sum    = 0.0
        absrel_sum  = 0.0
        delta1_sum  = 0.0
        n = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            depths = batch["depths"].to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                pred = self.model(images).squeeze(1)  # (B, H, W)

            m = self._depth_metrics(pred, depths)
            b = images.size(0)

            rmse_sum   += m["rmse"]   * b
            absrel_sum += m["abs_rel"] * b
            delta1_sum += m["delta1"] * b
            n          += b

        totals = torch.tensor(
            [rmse_sum, absrel_sum, delta1_sum, n],
            device=self.device,
            dtype=torch.float64,
        )
        totals = reduce_tensor_sum(totals)

        rmse_sum, absrel_sum, delta1_sum, n = totals.tolist()

        return {
            "rmse":    rmse_sum   / n,
            "abs_rel": absrel_sum / n,
            "delta1":  delta1_sum / n,
        }

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    def _silog_loss(
        self,
        pred:   torch.Tensor,   # (B, H, W)
        target: torch.Tensor,   # (B, H, W)
    ) -> torch.Tensor:
        """Scale-Invariant Log loss."""
        mask = (target > 0) & (pred > 1e-6)
        if mask.sum() == 0:
            return pred.sum() * 0.0  # zero loss, keeps graph alive

        d = torch.log(pred[mask]) - torch.log(target[mask])
        variance = d.var()
        mean_sq  = d.mean() ** 2
        return torch.sqrt(variance + (1.0 - self.SILOG_LAMBDA) * mean_sq + 1e-8)

    @staticmethod
    def _depth_metrics(
        pred:   torch.Tensor,   # (B, H, W)
        target: torch.Tensor,   # (B, H, W)
    ) -> dict:
        mask = (target > 0) & (pred > 1e-6)
        if mask.sum() == 0:
            return {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}

        p = pred[mask]
        t = target[mask]

        rmse    = torch.sqrt(((p - t) ** 2).mean()).item()
        abs_rel = (torch.abs(p - t) / t).mean().item()
        ratio   = torch.max(p / t, t / p)
        delta1  = (ratio < 1.25).float().mean().item()

        return {"rmse": rmse, "abs_rel": abs_rel, "delta1": delta1}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(self, epoch: int, rmse: float) -> None:
        state = {
            "epoch":      epoch + 1,
            "model":      unwrap_model(self.model).state_dict(),
            "optimiser":  self.optimiser.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "best_rmse":  self.best_rmse,
        }
        path = self.save_dir / f"epoch_{epoch+1:03d}.pth"
        torch.save(state, path)

        if rmse < self.best_rmse:
            self.best_rmse = rmse
            best_path = self.save_dir / "best.pth"
            torch.save(state, best_path)
            self.logger.info(f"  ✓ New best RMSE {rmse:.4f} — saved {best_path}")

    def _resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        unwrap_model(self.model).load_state_dict(ckpt["model"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"]
        self.best_rmse   = ckpt.get("best_rmse", float("inf"))
        self.logger.info(f"Resumed from {path} (epoch {self.start_epoch})")
