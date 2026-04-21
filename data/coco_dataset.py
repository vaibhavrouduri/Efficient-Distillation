"""
COCO Dataset Loader for Knowledge Distillation.

Returns batches of:
    {
        "images":  (B, 3, H, W)  float32 tensor,
        "targets": list of dicts with keys:
                       "boxes"   : (N, 4) xyxy float32
                       "labels"  : (N,)   int64
                       "image_id": int
    }

Requires:
    pip install pycocotools
    torchvision >= 0.13

COCO directory layout expected:
    <root>/
        annotations/
            instances_train2017.json
            instances_val2017.json
        train2017/
        val2017/
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_train_transforms(img_size: int = 224) -> Callable:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transforms(img_size: int = 224) -> Callable:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((img_size, img_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Collation helper
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Custom collate that handles variable-length annotation lists.
    Returns a dict with:
        "images"  : (B, 3, H, W)
        "targets" : list[dict]
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return {"images": images, "targets": list(targets)}


# ---------------------------------------------------------------------------
# Wrapper to convert COCO annotation format
# ---------------------------------------------------------------------------

class CocoDistillationDataset(CocoDetection):
    """
    Thin wrapper around torchvision CocoDetection that:
      - Applies transforms.
      - Converts raw COCO annotations to a clean dict format.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Optional[Callable] = None,
    ):
        # Pass transforms=None to parent; we apply them manually after
        # converting annotations so they operate on image+boxes together.
        super().__init__(root=root, annFile=annFile)
        self._transforms = transforms

    def __getitem__(self, idx: int):
        img, anns = super().__getitem__(idx)

        # Build target dict
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])   # convert to xyxy
            labels.append(ann["category_id"])

        target = {
            "boxes":    torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels":   torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([self.ids[idx]]),
        }

        if self._transforms is not None:
            img = self._transforms(img)

        return img, target


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_coco_dataloaders(
    coco_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build COCO train and val DataLoaders.

    Args:
        coco_root  : Path to the COCO root directory.
        img_size   : Spatial size to resize images to.
        batch_size : Batch size per GPU.
        num_workers: Number of DataLoader worker processes.
        pin_memory : Whether to pin memory (recommended when using CUDA).

    Returns:
        (train_loader, val_loader)
    """
    train_ann = os.path.join(coco_root, "annotations", "instances_train2017.json")
    val_ann   = os.path.join(coco_root, "annotations", "instances_val2017.json")
    train_img = os.path.join(coco_root, "train2017")
    val_img   = os.path.join(coco_root, "val2017")

    train_dataset = CocoDistillationDataset(
        root=train_img,
        annFile=train_ann,
        transforms=build_train_transforms(img_size),
    )
    val_dataset = CocoDistillationDataset(
        root=val_img,
        annFile=val_ann,
        transforms=build_val_transforms(img_size),
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        if distributed
        else None
    )

    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
