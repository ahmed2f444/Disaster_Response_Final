"""Portion 1 — Refactored xBD Training Utilities (Library Module, Pretrained Weights Active)."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from damage.dataset import XBDTileDataset, build_label_mask, list_tiles, load_xbd_footprints, tile_paths
from damage.model import CLASS_INDEX_TO_CANONICAL, DEFAULT_WEIGHTS_PATH, IGNORE_INDEX, NUM_CLASSES, build_model, resolve_device


@dataclass(slots=True)
class TrainConfig:
    """Configuration for optional xBD training or fine-tuning runs."""

    xbd_split_dir: Path
    train_disaster: str = "hurricane-matthew"
    val_disaster: str = "mexico-earthquake"
    train_tiles: int = 100
    val_tiles: int = 20
    tile_size: int = 1024
    crop_size: int = 256
    batch_size: int = 8
    learning_rate: float = 1e-3
    train_minutes: float = 20.0
    seed: int = 42
    model_path: Path = DEFAULT_WEIGHTS_PATH
    log_path: Path = Path("data/models/training_log.csv")
    class_count_path: Path = Path("data/models/class_counts.txt")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_classes(tile_ids: list[str], split_dir: Path, tile_size: int = 1024) -> np.ndarray:
    """Count per-class ground-truth pixel frequencies across the training tile list."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for tid in tile_ids:
        _, _, label_json = tile_paths(tid, split_dir)
        y = build_label_mask(load_xbd_footprints(label_json), (tile_size, tile_size))
        counts += np.bincount(y.ravel(), minlength=256)[:NUM_CLASSES]
    return counts


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, dict[int, float]]:
    """Evaluate validation loss and per-class building pixel accuracy."""
    model.eval()
    total_loss, batches = 0.0, 0
    correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    use_cuda = device == "cuda"

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", enabled=use_cuda):
            logits = model(x)
        loss = criterion(logits.float(), y)
        total_loss += float(loss.item())
        batches += 1
        pred = logits.argmax(dim=1)
        for c in range(1, NUM_CLASSES):
            mask = y == c
            totals[c] += int(mask.sum().item())
            correct[c] += int(((pred == c) & mask).sum().item())

    overall = float(correct[1:].sum() / max(int(totals[1:].sum()), 1))
    per_class = {
        c: (float(correct[c] / totals[c]) if totals[c] > 0 else float("nan"))
        for c in range(1, NUM_CLASSES)
    }
    return total_loss / max(batches, 1), overall, per_class


def run_training(cfg: TrainConfig) -> Path:
    """Run time-budgeted xBD segmentation training and persist the best checkpoint."""
    set_seed(cfg.seed)
    device = resolve_device()
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)

    train_ids = list_tiles(cfg.train_disaster, cfg.train_tiles, cfg.xbd_split_dir)
    val_ids = list_tiles(cfg.val_disaster, cfg.val_tiles, cfg.xbd_split_dir)

    counts = count_classes(train_ids, cfg.xbd_split_dir, cfg.tile_size)
    with open(cfg.class_count_path, "w", encoding="utf-8") as f:
        for c in range(NUM_CLASSES):
            name = "background" if c == 0 else CLASS_INDEX_TO_CANONICAL[c]
            f.write(f"{name}: {counts[c]}\n")

    train_ds = XBDTileDataset(train_ids, split_dir=cfg.xbd_split_dir, crop=cfg.crop_size, augment=True)
    val_ds = XBDTileDataset(val_ids, split_dir=cfg.xbd_split_dir, crop=None, augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=(device == "cuda"), drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(pretrained_encoder=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    budget = cfg.train_minutes * 60.0
    start = time.time()
    best_acc = -1.0

    with open(cfg.log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "elapsed_s", "train_loss", "val_loss", "val_acc_building_px", "acc_no_damage", "acc_minor", "acc_major", "acc_destroyed"])
        epoch = 0
        while time.time() - start < budget:
            epoch += 1
            model.train()
            running, steps = 0.0, 0
            for x, y in train_dl:
                if time.time() - start >= budget:
                    break
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits = model(x)
                    loss = criterion(logits.float(), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.item())
                steps += 1

            train_loss = running / max(steps, 1)
            val_loss, val_acc, per_class = evaluate(model, val_dl, criterion, device)
            elapsed = time.time() - start
            writer.writerow([
                epoch,
                round(elapsed, 1),
                round(train_loss, 4),
                round(val_loss, 4),
                round(val_acc, 4),
                *[round(per_class[c], 4) if per_class[c] == per_class[c] else "" for c in range(1, NUM_CLASSES)],
            ])
            f.flush()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), cfg.model_path)

    return cfg.model_path

