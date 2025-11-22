from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import timm


def _strip_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        cleaned[k] = v
    return cleaned


def load_dinov3_teacher(checkpoint_path: str, device: torch.device) -> nn.Module:
    teacher = timm.create_model(
        "vit_small_patch16_224", # teacher model name
        img_size=224,
        num_classes=1000,
    )
    raw = torch.load(checkpoint_path, map_location="cpu")
    state = raw.get("model", raw.get("state_dict", raw))
    state = _strip_prefix(state)
    teacher.load_state_dict(state, strict=False)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def build_student(num_classes: int, device: torch.device) -> nn.Module:
    student = timm.create_model(
        "vit_tiny_patch16_224", # student model name
        img_size=224,
        num_classes=num_classes,
    )
    return student.to(device)


@dataclass
class DistillConfig:
    dataset_path: str = str(Path(" ") / " ") # Provide path to your dataset
    teacher_ckpt: str = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    output_path: str = "student_dinov3_kl.pth"

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.05
    num_workers: int = 0
    temperature: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_split: float = 0.1
    warmup_ratio: float = 0.05
    seed: int = 42


def _gather_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    root_path = Path(root)
    files = [
        str(p)
        for p in root_path.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    return files


def make_dataloaders(cfg: DistillConfig):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    image_paths = _gather_images(cfg.dataset_path)

    class ImageOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img)

    rng = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(image_paths), generator=rng).tolist()
    val_size = max(1, int(len(indices) * cfg.val_split))
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    def make_loader(idxs, shuffle):
        subset = [image_paths[i] for i in idxs]
        return DataLoader(
            ImageOnlyDataset(subset),
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return make_loader(train_idx, True), make_loader(val_idx, False)


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    t = temperature
    return (
        F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
        )
        * (t * t)
    )


def distill(cfg: DistillConfig) -> None:
    device = torch.device(cfg.device)
    teacher = load_dinov3_teacher(cfg.teacher_ckpt, device)
    student = build_student(num_classes=teacher.num_classes, device=device)
    train_loader, val_loader = make_dataloaders(cfg)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    print(
        f"train_images={len(train_loader.dataset)} | val_images={len(val_loader.dataset)} | "
        f"batch_size={cfg.batch_size} | epochs={cfg.epochs} | lr={cfg.lr} | "
        f"weight_decay={cfg.weight_decay} | temp={cfg.temperature}",
        flush=True,
    )

    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(cfg.epochs):
        student.train()
        running_loss = 0.0
        for images in train_loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = kl_distillation_loss(student_logits, teacher_logits, cfg.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
        train_loss = running_loss / max(1, len(train_loader))

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device, non_blocking=True)
                teacher_logits = teacher(images)
                student_logits = student(images)
                val_loss += kl_distillation_loss(
                    student_logits, teacher_logits, cfg.temperature
                ).item()
        val_loss = val_loss / max(1, len(val_loader))

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in student.state_dict().items()
            }
            best_epoch = epoch + 1
            improved = " (best)"

        print(
            f"[epoch {epoch+1}/{cfg.epochs}] train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f}{improved}",
            flush=True,
        )

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    torch.save(
        {
            "student_state_dict": best_state or student.state_dict(),
            "config": cfg.__dict__,
            "teacher_checkpoint": cfg.teacher_ckpt,
        },
        cfg.output_path,
    )
    print(
        f"Saved student weights to {cfg.output_path} "
        f"(best_epoch={best_epoch}, best_val_loss={best_val:.4f})"
    )


def parse_args() -> DistillConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",      dest="dataset_path",  type=str)
    parser.add_argument("--teacher-ckpt", dest="teacher_ckpt",  type=str)
    parser.add_argument("--out",          dest="output_path",   type=str)

    parser.add_argument("--batch-size",   dest="batch_size",    type=int)
    parser.add_argument("--epochs",       dest="epochs",        type=int)
    parser.add_argument("--lr",           dest="lr",            type=float)
    parser.add_argument("--weight-decay", dest="weight_decay",  type=float)
    parser.add_argument("--num-workers",  dest="num_workers",   type=int)
    parser.add_argument("--temperature",  dest="temperature",   type=float)
    parser.add_argument("--device",       dest="device",        type=str)
    parser.add_argument("--val-split",    dest="val_split",     type=float)
    parser.add_argument("--warmup-ratio", dest="warmup_ratio",  type=float)
    parser.add_argument("--seed",         dest="seed",          type=int)

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    cfg = DistillConfig(**overrides)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    distill(cfg)
