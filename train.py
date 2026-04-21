"""
Main entry-point for training.

Dispatches on cfg["task"]:
  "imagenet"   — Swin teacher→student knowledge distillation (original pipeline)
  "depth"      — DPT depth head on frozen distilled student, NYU-Depth V2
  "lora_depth" — LoRA fine-tuning of frozen student+DPT on NYU-Depth V2

Usage:
    python train.py --config configs/distill_config.yaml
    python train.py --config configs/depth_config.yaml
    python train.py --config configs/lora_depth_config.yaml
    python train.py --config configs/distill_config.yaml --resume checkpoints/epoch_010.pth
"""

from __future__ import annotations

import argparse
import yaml
import torch

from utils.logger import setup_logger
from utils.wandb_logger import init_wandb
from utils.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    wrap_ddp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Efficient Distillation Training")
    parser.add_argument("--config", type=str, default="configs/distill_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from.")
    parser.add_argument("--adapter_warm_start", type=str, default=None,
                        help="Path to a checkpoint whose adapter weights are used to "
                             "warm-start the adapters only (student & optimiser stay fresh). "
                             "Ignored when --resume is also provided.")
    parser.add_argument("--override", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="Override config values, e.g. --override lr=1e-4 epochs=50")
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply KEY=VALUE overrides to a config dict, casting to int/float/bool where appropriate."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE format.")
        key, raw = item.split("=", 1)
        for cast in (int, float):
            try:
                raw = cast(raw); break
            except ValueError:
                pass
        else:
            if raw.lower() in ("true", "false"):
                raw = raw.lower() == "true"
        cfg[key] = raw
    return cfg


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ #
# Task: ImageNet distillation (original pipeline, unchanged)
# ------------------------------------------------------------------ #

def _run_imagenet(cfg: dict, device: torch.device, wandb_run, logger) -> None:
    from models.teacher import SwinTeacher
    from models.student import SwinStudentTiny
    from models.adapters import FeatureAdapter
    from distillation.losses import DistillationLoss
    from distillation.trainer import DistillationTrainer
    from data.imagenet_dataset import build_imagenet_dataloaders

    train_loader, val_loader = build_imagenet_dataloaders(
        imagenet_root = cfg["imagenet_root"],
        img_size = cfg.get("img_size", 224),
        batch_size = cfg.get("batch_size", 32),
        num_workers = cfg.get("num_workers", 8),
        pin_memory = cfg.get("pin_memory", True),
        distributed = cfg.get("distributed", False),
        rank = cfg.get("rank", 0),
        world_size = cfg.get("world_size", 1),
    )
    logger.info(
        f"ImageNet  train: {len(train_loader.dataset):,}  |  "
        f"val: {len(val_loader.dataset):,}"
    )

    teacher_variant = cfg.get("teacher_variant", "swin_large")
    teacher = SwinTeacher(
        variant       = teacher_variant,
        pretrained    = cfg.get("teacher_pretrained", True),
        num_classes   = cfg.get("num_classes", 1000),
        frozen_stages = cfg.get("teacher_frozen_stages", 4),
    )
    student = SwinStudentTiny(
        pretrained  = cfg.get("student_pretrained", True),
        num_classes = cfg.get("num_classes", 1000),
    )
    logger.info(
        f"Teacher ({teacher_variant})  params: {teacher.num_parameters:,}  "
        f"trainable: {teacher.num_trainable_parameters:,}"
    )
    logger.info(
        f"Student (swin_tiny)  params: {student.num_parameters:,}  "
        f"trainable: {student.num_trainable_parameters:,}"
    )

    adapter = FeatureAdapter(
        student_channels  = student.stage_channels,
        teacher_channels  = teacher.stage_channels,
        stages            = cfg.get("adapter_stages", [0, 1, 2, 3]),
        use_spatial_align = cfg.get("adapter_spatial_align", True),
    )
    logger.info(f"Adapter params: {adapter.num_parameters:,}")

    use_compile = cfg.get("compile", True) and not cfg.get("distributed", False)

    if use_compile and hasattr(torch, "compile"):
        try:
            import triton  # noqa: F401
            logger.info("torch.compile: student + adapter ...")
            student = torch.compile(student)
            adapter = torch.compile(adapter)
            logger.info("torch.compile done.")
        except ImportError:
            logger.warning("triton not found — skipping torch.compile.")

    loss_fn = DistillationLoss(
        w_feat         = cfg.get("w_feat", 1.0),
        w_at           = cfg.get("w_at",   0.5),
        w_kd           = cfg.get("w_kd",   1.0),
        w_task         = cfg.get("w_task",  1.0),
        temperature    = cfg.get("temperature", 4.0),
        feat_loss_type = cfg.get("feat_loss_type", "mse"),
    )

    trainer = DistillationTrainer(
        teacher      = teacher,
        student      = student,
        adapter      = adapter,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
        wandb_run    = wandb_run,
    )
    trainer.train()


# ------------------------------------------------------------------ #
# Task: Depth estimation — frozen student backbone + DPT head
# ------------------------------------------------------------------ #

def _run_depth(cfg: dict, device: torch.device, wandb_run, logger) -> None:
    from models.depth_model import StudentWithDPT
    from distillation.depth_trainer import DepthTrainer
    from data.nyu_depth_dataset import build_nyu_depth_dataloaders

    train_loader, val_loader = build_nyu_depth_dataloaders(
        root = cfg["nyu_depth_root"],
        img_size = cfg.get("img_size", 224),
        batch_size = cfg.get("batch_size", 16),
        num_workers = cfg.get("num_workers", 8),
        pin_memory = cfg.get("pin_memory", True),
        distributed = cfg.get("distributed", False),
        rank = cfg.get("rank", 0),
        world_size = cfg.get("world_size", 1),
    )
    logger.info(
        f"NYU-Depth  train: {len(train_loader.dataset):,}  |  "
        f"val: {len(val_loader.dataset):,}"
    )

    model = StudentWithDPT(
        student_checkpoint = cfg["student_checkpoint"],
        dpt_pretrained     = cfg.get("dpt_pretrained", True),
        freeze_student     = cfg.get("freeze_student", True),
    )
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"StudentWithDPT  total: {total:,}  trainable: {trainable:,}")

    trainer = DepthTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
        wandb_run    = wandb_run,
    )
    trainer.train()


# ------------------------------------------------------------------ #
# Task: LoRA depth — LoRA on frozen student+DPT
# ------------------------------------------------------------------ #

def _run_lora_depth(cfg: dict, device: torch.device, wandb_run, logger) -> None:
    from models.depth_model import StudentWithDPT
    from models.lora import apply_lora, count_lora_params
    from distillation.lora_depth_trainer import LoRADepthTrainer
    from data.nyu_depth_dataset import build_nyu_depth_dataloaders

    train_loader, val_loader = build_nyu_depth_dataloaders(
        root = cfg["nyu_depth_root"],
        img_size = cfg.get("img_size", 224),
        batch_size = cfg.get("batch_size", 16),
        num_workers = cfg.get("num_workers", 8),
        pin_memory = cfg.get("pin_memory", True),
        distributed = cfg.get("distributed", False),
        rank = cfg.get("rank", 0),
        world_size = cfg.get("world_size", 1),
    )
    logger.info(
        f"NYU-Depth  train: {len(train_loader.dataset):,}  |  "
        f"val: {len(val_loader.dataset):,}"
    )

    # Load pretrained student+DPT from depth training checkpoint
    model = StudentWithDPT(
        student_checkpoint = cfg["student_checkpoint"],
        dpt_pretrained     = cfg.get("dpt_pretrained", True),
        freeze_student     = True,
    )

    # Optionally load depth checkpoint (student+DPT already trained)
    if cfg.get("depth_checkpoint"):
        ckpt = torch.load(cfg["depth_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"], strict=True)
        logger.info(f"Loaded depth checkpoint: {cfg['depth_checkpoint']}")

    # Freeze everything, then apply LoRA to student attention layers
    for p in model.parameters():
        p.requires_grad = False

    apply_lora(
        model.student,
        r               = cfg.get("lora_r", 4),
        alpha           = cfg.get("lora_alpha", 1.0),
        target_modules  = cfg.get("lora_target_modules", ["qkv"]),
    )
    # DPT head: trainable by default, optionally frozen (LoRA-only mode)
    freeze_dpt = cfg.get("freeze_dpt_head", False)
    for p in model.dpt_head.parameters():
        p.requires_grad = not freeze_dpt
    if freeze_dpt:
        logger.info("DPT head frozen — training LoRA adapters only.")

    lora_params = count_lora_params(model.student)
    dpt_params  = sum(p.numel() for p in model.dpt_head.parameters())
    logger.info(f"LoRA params: {lora_params:,}  |  DPT head params: {dpt_params:,}")

    trainer = LoRADepthTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
        wandb_run    = wandb_run,
    )
    trainer.train()


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    cfg    = apply_overrides(cfg, args.override)
    logger = setup_logger("train")

    if args.resume:
        cfg["resume"] = args.resume
    if args.adapter_warm_start and not cfg.get("resume"):
        cfg["adapter_warm_start"] = args.adapter_warm_start

    if cfg.get("smoke_test_batches", 0):
        cfg.setdefault("wandb", {})["enabled"] = False
        logger.info("smoke_test_batches set — W&B disabled for this run")

    if args.override and cfg.get("wandb", {}).get("run_name") is None:
        tag = "_".join(o.replace("=", "") for o in args.override)
        cfg.setdefault("wandb", {})["run_name"] = tag

    dist_ctx = setup_distributed()
    device = dist_ctx["device"]

    cfg["distributed"] = dist_ctx["distributed"]
    cfg["rank"] = dist_ctx["rank"]
    cfg["local_rank"] = dist_ctx["local_rank"]
    cfg["world_size"] = dist_ctx["world_size"]

    wandb_run = init_wandb(cfg) if is_main_process() else None

    logger.info(
        f"distributed={cfg['distributed']} "
        f"rank={cfg['rank']} "
        f"local_rank={cfg['local_rank']} "
        f"world_size={cfg['world_size']} "
        f"device={device}"
    )

    task = cfg.get("task", "imagenet")
    logger.info(f"Task: {task}")

    try:
        if task == "imagenet":
            _run_imagenet(cfg, device, wandb_run, logger)
        elif task == "depth":
            _run_depth(cfg, device, wandb_run, logger)
        elif task == "lora_depth":
            _run_lora_depth(cfg, device, wandb_run, logger)
        else:
            raise ValueError(
                f"Unknown task: {task!r}. "
                "Choose 'imagenet', 'depth', or 'lora_depth'."
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
