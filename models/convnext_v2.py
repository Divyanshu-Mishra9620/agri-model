"""ConvNeXt V2 Tiny builder via timm, with a dynamically-sized classifier head."""

from __future__ import annotations

import logging

import timm
import torch.nn as nn

from configs.schema import ModelConfig

logger = logging.getLogger(__name__)


def build_model(num_classes: int, cfg: ModelConfig) -> nn.Module:
    """Create a timm ConvNeXt V2 Tiny with `num_classes` outputs.

    timm's `create_model(..., num_classes=...)` already replaces the
    classifier head for us — this is the robust way to do it across timm's
    ConvNeXt V2 variants, since the exact head submodule name/shape isn't
    part of timm's public API contract and can differ between model tags.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be a positive, dataset-derived value, got {num_classes}")

    model = timm.create_model(
        cfg.name,
        pretrained=cfg.pretrained,
        num_classes=num_classes,
        drop_path_rate=cfg.drop_path_rate,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built %s (pretrained=%s) with %d classes — %.1fM parameters.",
        cfg.name, cfg.pretrained, num_classes, n_params / 1e6,
    )
    return model


def get_classifier_param_ids(model: nn.Module) -> set[int]:
    """Identify which parameters belong to the classifier head via timm's
    `get_classifier()` (a stable, public part of timm's API contract),
    rather than guessing a submodule name like `.head` or `.fc` that
    differs across timm's ConvNeXt V2 tags. Returns `id(param)` values —
    the standard way to build a set-membership test against `nn.Parameter`
    objects, which are not hashable by value.

    Used by both `engine.optimizer.build_optimizer` (discriminative LR) and
    `freeze_backbone` below (Stage 2 decoupled fine-tuning), so head
    identification lives in exactly one place.
    """
    classifier = model.get_classifier()
    return {id(p) for p in classifier.parameters()}


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the classifier head, for Stage 2
    decoupled classifier re-training (see configs/schema.py::Stage2Config).
    """
    head_param_ids = get_classifier_param_ids(model)
    n_frozen, n_trainable = 0, 0
    for param in model.parameters():
        if id(param) in head_param_ids:
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False
            n_frozen += param.numel()

    logger.info(
        "Froze backbone for Stage 2: %.1fM params frozen, %.2fM params (classifier head) trainable.",
        n_frozen / 1e6, n_trainable / 1e6,
    )
