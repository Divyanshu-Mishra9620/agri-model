from __future__ import annotations

import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    LRScheduler,
    OneCycleLR,
    SequentialLR,
)

from configs.schema import SchedulerConfig

logger = logging.getLogger(__name__)

def build_scheduler(
    optimizer: Optimizer, cfg: SchedulerConfig, *, epochs: int, steps_per_epoch: int
) -> LRScheduler:
    if cfg.name == "cosine":
        scheduler = _cosine_with_warmup(optimizer, cfg, epochs=epochs)
        scheduler.step_every = "epoch"
    elif cfg.name == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.warm_restarts_t0, T_mult=cfg.warm_restarts_tmult, eta_min=cfg.min_lr,
        )
        scheduler.step_every = "epoch"
    elif cfg.name == "onecycle":
        max_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lrs, epochs=epochs, steps_per_epoch=steps_per_epoch,
            pct_start=cfg.onecycle_pct_start,
        )
        scheduler.step_every = "batch"
    else:
        raise ValueError(f"Unknown scheduler.name: {cfg.name!r}")

    logger.info("Built '%s' scheduler (steps every %s).", cfg.name, scheduler.step_every)
    return scheduler

def _cosine_with_warmup(optimizer: Optimizer, cfg: SchedulerConfig, *, epochs: int) -> LRScheduler:
    warmup_epochs = max(0, min(cfg.warmup_epochs, epochs - 1))

    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=cfg.min_lr)

    warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=cfg.min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
