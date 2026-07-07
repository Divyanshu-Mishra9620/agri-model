from __future__ import annotations

import logging

import torch.nn as nn
from torch.optim import AdamW, Optimizer

from configs.schema import OptimizerConfig
from models.convnext_v2 import get_classifier_param_ids

logger = logging.getLogger(__name__)

def _build_param_groups(model: nn.Module, cfg: OptimizerConfig, apply_head_lr_multiplier: bool = True) -> list[dict]:

    head_param_ids = get_classifier_param_ids(model)
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = id(param) in head_param_ids
        no_decay = cfg.no_weight_decay_on_norm_and_bias and (param.ndim <= 1 or name.endswith(".bias"))
        groups.setdefault((is_head, no_decay), []).append(param)

    param_groups = []
    for (is_head, no_decay), params in groups.items():
        lr = cfg.lr * cfg.head_lr_multiplier if (is_head and apply_head_lr_multiplier) else cfg.lr
        weight_decay = 0.0 if no_decay else cfg.weight_decay
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
        logger.info(
            "AdamW param group: %s/%s — %d tensors, lr=%.2e, weight_decay=%.3f",
            "head" if is_head else "backbone", "no_decay" if no_decay else "decay",
            len(params), lr, weight_decay,
        )

    if not param_groups:
        raise ValueError("No trainable parameters found")

    return param_groups

def build_optimizer(model: nn.Module, cfg: OptimizerConfig, apply_head_lr_multiplier: bool = True) -> Optimizer:
    if cfg.name != "adamw":
        raise ValueError(f"Unsupported optimizer.name: {cfg.name!r} (only 'adamw' is implemented)")

    param_groups = _build_param_groups(model, cfg, apply_head_lr_multiplier=apply_head_lr_multiplier)
    return AdamW(param_groups, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
