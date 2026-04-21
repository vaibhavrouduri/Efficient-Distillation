"""
Distillation Trainer.

Orchestrates the full training loop:
    1. Forward pass through the frozen/partially-frozen teacher.
    2. Forward pass through the student.
    3. Adapter projection of student features.
    4. Compute combined distillation + task loss.
    5. Backward pass and optimiser step.
    6. LR-scheduler step.
    7. Logging & checkpointing.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.teacher import SwinTeacher
from models.student import SwinStudentTiny
from models.adapters import FeatureAdapter
from distillation.losses import DistillationLoss
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_metrics
from utils.distributed import is_main_process, reduce_mean, wrap_ddp


class DistillationTrainer:
    """
    Full distillation training loop.

    Args:
        teacher       : Pretrained :class:`SwinTeacher`.
        student       : :class:`SwinStudentTiny` to be trained.
        adapter       : :class:`FeatureAdapter` bridging the two backbones.
        loss_fn       : :class:`DistillationLoss` instance.
        train_loader  : DataLoader for COCO training split.
        val_loader    : DataLoader for COCO validation split.
        cfg           : Flat config dict (see ``configs/distill_config.yaml``).
        device        : torch.device.
    """

    def __init__(
        self,
        teacher: SwinTeacher,
        student: SwinStudentTiny,
        adapter: FeatureAdapter,
        loss_fn: DistillationLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        wandb_run=None,
    ):
        self.teacher      = teacher.to(device)
        self.student      = student.to(device)
        self.adapter      = adapter.to(device)

        if self.distributed:
            self.student = wrap_ddp(self.student, device)
            self.adapter = wrap_ddp(self.adapter, device)
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.logger       = setup_logger("DistillationTrainer")
        self.wandb_run    = wandb_run
        self.is_main_process = is_main_process()
        self.distributed = cfg.get("distributed", False)

        # Teacher is always in eval mode — we only distil from it
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Optimiser: only student + adapter parameters
        trainable_params = list(self.student.parameters()) + list(self.adapter.parameters())
        self.optimiser = torch.optim.AdamW(
            trainable_params,
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-2),
        )

        # Cosine LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser,
            T_max=cfg.get("epochs", 30),
            eta_min=cfg.get("lr_min", 1e-6),
        )

        # Mixed-precision scaler
        self.scaler = GradScaler("cuda", enabled=cfg.get("amp", True))

        self.start_epoch = 0
        self.best_metric = 0.0

        # Resume if checkpoint exists
        ckpt_path = cfg.get("resume", None)
        if ckpt_path and os.path.isfile(ckpt_path):
            self.start_epoch, self.best_metric = load_checkpoint(
                ckpt_path, self.student, self.adapter, self.optimiser, self.scheduler
            )
            self.logger.info(f"Resumed from checkpoint: {ckpt_path}")
        else:
            # Warm-start adapters only from a previous checkpoint.
            # Used when student_pretrained=False: the randomly-initialised student
            # produces feature maps with an arbitrary scale/distribution on step 0,
            # which causes the adapter outputs to be near-zero-norm vectors →
            # cosine loss is undefined → NaN in loss_feat.
            # Loading previously-trained adapter weights gives the projections a
            # sensible starting point without touching the student, optimiser, or
            # scheduler — so training still proceeds from epoch 0 with a fresh student.
            warm_start_path = cfg.get("adapter_warm_start", None)
            if warm_start_path and os.path.isfile(warm_start_path):
                self._load_adapter_warm_start(warm_start_path)

    def _load_adapter_warm_start(self, path: str) -> None:
        """Load **only** the adapter weights from ``path``.

        The student, optimiser, scheduler, and epoch counter are left untouched
        so that training starts fresh from epoch 0 with a randomly-initialised
        student and warm-started adapter projections.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "adapter" not in ckpt:
            self.logger.warning(
                f"[adapter_warm_start] No 'adapter' key found in {path} — skipping."
            )
            return

        state_dict = ckpt["adapter"]

        # Checkpoints saved while torch.compile was active store keys prefixed
        # with '_orig_mod.' (e.g. '_orig_mod.adapters.stage_0.proj.0.weight').
        # The live adapter at warm-start time is the unwrapped module (compile
        # happens after __init__), so strip that prefix if present.
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
            }

        # Handle torch.compile wrapping on the live model side just in case
        adapter_target = (
            self.adapter._orig_mod
            if hasattr(self.adapter, "_orig_mod")
            else self.adapter
        )
        missing, unexpected = adapter_target.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.warning(
                f"[adapter_warm_start] Missing keys when loading adapter: {missing}"
            )
        if unexpected:
            self.logger.warning(
                f"[adapter_warm_start] Unexpected keys when loading adapter: {unexpected}"
            )
        src_epoch = ckpt.get("epoch", "?")
        self.logger.info(
            f"[adapter_warm_start] Loaded adapter weights from '{path}' "
            f"(source epoch {src_epoch}). Student & optimiser remain freshly initialised."
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop."""
        epochs = self.cfg.get("epochs", 30)
        save_dir = self.cfg.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(self.start_epoch, epochs):
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_losses = self._train_one_epoch(epoch)
            val_metrics  = self._validate(epoch)

            self.scheduler.step()

            # Log epoch summary
            self.logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {train_losses['total']:.4f} "
                f"(feat={train_losses['feat']:.4f}, "
                f"at={train_losses['at']:.4f}, "
                f"kd={train_losses['kd']:.4f}) | "
                f"Student Top-1: {val_metrics.get('val_student_top1', 0.0):.4f}  "
                f"Top-5: {val_metrics.get('val_student_top5', 0.0):.4f}  "
                f"(Teacher: {val_metrics.get('val_teacher_top1', 0.0):.4f})"
            )

            if self.wandb_run is not None and self.is_main_process:
                epoch_log = {
                    "epoch": epoch + 1,
                    "epoch/train_loss_total": train_losses["total"],
                    "epoch/train_loss_feat":  train_losses["feat"],
                    "epoch/train_loss_at":    train_losses["at"],
                    "epoch/train_loss_kd":    train_losses["kd"],
                    "epoch/val_loss_feat":    val_metrics.get("val_feat",  0.0),
                    "epoch/val_loss_at":      val_metrics.get("val_at",    0.0),
                    "epoch/val_loss_kd":      val_metrics.get("val_kd",    0.0),
                    "epoch/val_student_top1": val_metrics.get("val_student_top1", 0.0),
                    "epoch/val_student_top5": val_metrics.get("val_student_top5", 0.0),
                    "epoch/val_teacher_top1": val_metrics.get("val_teacher_top1", 0.0),
                    "epoch/val_teacher_top5": val_metrics.get("val_teacher_top5", 0.0),
                    "epoch/lr":               self.optimiser.param_groups[0]["lr"],
                }
                self.wandb_run.log(epoch_log, step=(epoch + 1) * len(self.train_loader))

            # Save best checkpoint
            current_metric = val_metrics.get("mAP", 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            if self.is_main_process:
                save_checkpoint(
                    path=os.path.join(save_dir, f"epoch_{epoch+1:03d}.pth"),
                    epoch=epoch + 1,
                    student=self.student,
                    adapter=self.adapter,
                    optimiser=self.optimiser,
                    scheduler=self.scheduler,
                    best_metric=self.best_metric,
                    is_best=is_best,
                )

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        self.student.train()
        self.adapter.train()
        self.teacher.eval()

        running = {k: 0.0 for k in ("total", "feat", "at", "kd", "task")}
        n_batches = len(self.train_loader)
        t0 = time.time()

        smoke = self.cfg.get("smoke_test_batches", 0)

        for batch_idx, batch in enumerate(self.train_loader):
            if smoke and batch_idx >= smoke:
                self.logger.info(f"  [smoke test] stopping after {smoke} batches")
                break

            images  = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)   # (B,) int64 — always available now

            self.optimiser.zero_grad()

            with autocast("cuda", enabled=self.cfg.get("amp", True)):
                # Teacher forward (no_grad – already set via requires_grad=False)
                with torch.no_grad():
                    t_feats, t_logits = self.teacher(images)

                # Student forward
                s_feats, s_logits = self.student(images)

                # Adapter projection
                adapted_s_feats = self.adapter(s_feats, t_feats)

                # Optional ground-truth cross-entropy task loss
                task_loss = None
                if self.cfg.get("use_gt_loss", False) and s_logits is not None:
                    task_loss = nn.functional.cross_entropy(s_logits, targets)

                # Compute losses
                loss_dict = self.loss_fn(
                    adapted_student_feats=adapted_s_feats,
                    teacher_feats=t_feats,
                    student_feats_raw=s_feats,
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    task_loss=task_loss,
                )

            self.scaler.scale(loss_dict["total"]).backward()
            self.scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.adapter.parameters()),
                max_norm=self.cfg.get("grad_clip", 1.0),
            )
            self.scaler.step(self.optimiser)
            self.scaler.update()

            for k in running:
                running[k] += loss_dict[k].item()

            if batch_idx % self.cfg.get("log_every", 50) == 0:
                elapsed = time.time() - t0
                self.logger.info(
                    f"  [Epoch {epoch+1} | {batch_idx}/{n_batches}] "
                    f"loss={loss_dict['total'].item():.4f}  "
                    f"({elapsed:.1f}s elapsed)"
                )
                if self.wandb_run is not None and self.is_main_process:
                    global_step = epoch * n_batches + batch_idx
                    self.wandb_run.log(
                        {
                            "train/loss_total": loss_dict["total"].item(),
                            "train/loss_feat":  loss_dict["feat"].item(),
                            "train/loss_at":    loss_dict["at"].item(),
                            "train/loss_kd":    loss_dict["kd"].item(),
                            "train/loss_task":  loss_dict["task"].item(),
                            "train/lr": self.optimiser.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

        averages = {k: v / n_batches for k, v in running.items()}
        averages = {k: reduce_mean(v, self.device) for k, v in averages.items()}
        return averages

    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validation pass – returns a dict of metrics."""
        self.student.eval()
        self.adapter.eval()

        smoke = self.cfg.get("smoke_test_batches", 0)
        s_preds, t_preds, all_targets = [], [], []
        running_val = {"feat": 0.0, "at": 0.0, "kd": 0.0}
        n_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if smoke and batch_idx >= smoke:
                    break

                images  = batch["images"].to(self.device)
                targets = batch["targets"]          # (B,) int64 — ImageNet class indices

                t_feats, t_logits = self.teacher(images)
                s_feats, s_logits = self.student(images)
                adapted_s_feats   = self.adapter(s_feats, t_feats)

                # Val distillation losses (unweighted, for monitoring)
                l_feat = self.loss_fn.feature_loss(adapted_s_feats, t_feats)
                l_at   = self.loss_fn.attention_transfer_loss(s_feats, t_feats)
                l_kd   = self.loss_fn.kd_loss(s_logits, t_logits) if (s_logits is not None and t_logits is not None) else torch.tensor(0.0)

                running_val["feat"] += l_feat.item()
                running_val["at"]   += l_at.item()
                running_val["kd"]   += l_kd.item()
                n_batches += 1

                if s_logits is not None:
                    s_preds.append(s_logits.cpu())
                if t_logits is not None:
                    t_preds.append(t_logits.cpu())
                all_targets.append(targets.cpu())

        metrics = {k: v / max(n_batches, 1) for k, v in running_val.items()}
        metrics = {f"val_{k}": v for k, v in metrics.items()}

        if all_targets:
            targets_cat = torch.cat(all_targets, dim=0)

            # Student accuracy
            if s_preds:
                s_acc = compute_metrics(torch.cat(s_preds, dim=0), targets_cat)
                metrics["val_student_top1"] = s_acc.get("top1", 0.0)
                metrics["val_student_top5"] = s_acc.get("top5", 0.0)
                metrics["mAP"] = s_acc.get("top1", 0.0)   # used for checkpoint best-metric

            # Teacher accuracy (ceiling reference)
            if t_preds:
                t_acc = compute_metrics(torch.cat(t_preds, dim=0), targets_cat)
                metrics["val_teacher_top1"] = t_acc.get("top1", 0.0)
                metrics["val_teacher_top5"] = t_acc.get("top5", 0.0)

        return metrics
