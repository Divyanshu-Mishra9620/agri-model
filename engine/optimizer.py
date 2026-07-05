"""Optimizer factory: AdamW with discriminative backbone/head learning rates
and configurable no-weight-decay parameter groups.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch.optim import AdamW, Optimizer

from configs.schema import OptimizerConfig
from models.convnext_v2 import get_classifier_param_ids

logger = logging.getLogger(__name__)


def _build_param_groups(model: nn.Module, cfg: OptimizerConfig) -> list[dict]:
    """Split trainable parameters into up to 4 groups: {backbone, head} x
    {decay, no_decay}. Frozen parameters (requires_grad=False, e.g. during a
    Stage 2 backbone-frozen fine-tune) are skipped entirely.

    Discriminative LR (always applied): the classifier head is freshly
    initialized while the backbone starts from strong pretraining, so the
    head gets `lr * head_lr_multiplier` and the backbone gets `lr` —
    otherwise the head's large early gradients can backpropagate into and
    distort good pretrained features (see OptimizerConfig.head_lr_multiplier).

    No-decay (applied when cfg.no_weight_decay_on_norm_and_bias): biases and
    any 1-D parameter (LayerNorm weight/bias) are excluded from weight
    decay — standard practice for ConvNeXt-family models, since decaying
    normalization scale/shift tends to hurt rather than usefully regularize.
    """
    head_param_ids = get_classifier_param_ids(model)
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}  # (is_head, no_decay) -> params

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = id(param) in head_param_ids
        no_decay = cfg.no_weight_decay_on_norm_and_bias and (param.ndim <= 1 or name.endswith(".bias"))
        groups.setdefault((is_head, no_decay), []).append(param)

    param_groups = []
    for (is_head, no_decay), params in groups.items():
        lr = cfg.lr * cfg.head_lr_multiplier if is_head else cfg.lr
        weight_decay = 0.0 if no_decay else cfg.weight_decay
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
        logger.info(
            "AdamW param group: %s/%s — %d tensors, lr=%.2e, weight_decay=%.3f",
            "head" if is_head else "backbone", "no_decay" if no_decay else "decay",
            len(params), lr, weight_decay,
        )

    if not param_groups:
        raise ValueError("No trainable parameters found — did you freeze everything by mistake?")

    return param_groups


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> Optimizer:
    """Build the configured optimizer (currently only AdamW is implemented,
    the standard choice for fine-tuning ConvNeXt-family models).

    `lr=cfg.lr` passed to AdamW itself is a required default but is never
    actually used for stepping, since every param group above supplies its
    own explicit `lr`.
    """
    if cfg.name != "adamw":
        raise ValueError(f"Unsupported optimizer.name: {cfg.name!r} (only 'adamw' is implemented)")

    param_groups = _build_param_groups(model, cfg)
    return AdamW(param_groups, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
