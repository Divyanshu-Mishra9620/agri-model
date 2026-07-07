from __future__ import annotations

import dataclasses
import logging
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from configs.schema import Config
from utils.device import unwrap_model

logger = logging.getLogger(__name__)

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    best_metric: float,
    class_to_idx: dict[str, int],
    config: Config,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "class_to_idx": class_to_idx,
        "config": dataclasses.asdict(config),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    logger.info("Saved checkpoint: %s (epoch=%d, best_metric=%.4f)", path, epoch, best_metric)

def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    logger.info("Loaded checkpoint: %s (epoch=%s)", path, checkpoint.get("epoch", "?"))
    return checkpoint

def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    restore_rng: bool = True,
) -> tuple[int, float]:
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if restore_rng and "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].to(torch.uint8).cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_metric = float(checkpoint.get("best_metric", float("-inf")))
    logger.info("Resumed from epoch %d (best_metric so far=%.4f)", start_epoch, best_metric)
    return start_epoch, best_metric
