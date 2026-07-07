from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import LossConfig

logger = logging.getLogger(__name__)

def _label_smoothed_nll(log_probs: torch.Tensor, targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return (1.0 - epsilon) * nll + epsilon * smooth

class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        per_sample_nll = _label_smoothed_nll(log_probs, targets, self.label_smoothing)

        p_t = log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1).exp()
        focal_weight = (1.0 - p_t).clamp(min=1e-6).pow(self.gamma)

        return (focal_weight * per_sample_nll).mean()

def _inverse_frequency_weight_tensor(
    class_counts: dict[str, int], class_to_idx: dict[str, int], power: float
) -> torch.Tensor:
    n_classes = len(class_to_idx)
    weights = torch.ones(n_classes, dtype=torch.float32)
    for name, idx in class_to_idx.items():
        count = class_counts.get(name, 1)
        weights[idx] = (1.0 / count) ** power
    return weights * (n_classes / weights.sum())

def build_loss(
    cfg: LossConfig,
    class_counts: Optional[dict[str, int]] = None,
    class_to_idx: Optional[dict[str, int]] = None,
) -> nn.Module:
    if cfg.name == "ce":
        return nn.CrossEntropyLoss()

    if cfg.name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if cfg.name == "focal":
        return FocalLoss(gamma=cfg.focal_gamma, label_smoothing=0.0)

    if cfg.name == "focal_ls":
        return FocalLoss(gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)

    if cfg.name == "weighted_ce":
        if class_counts is None or class_to_idx is None:
            raise ValueError("loss.name='weighted_ce' requires class_counts and class_to_idx.")
        weights = _inverse_frequency_weight_tensor(class_counts, class_to_idx, cfg.class_weight_power)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    raise ValueError(f"Unknown loss.name: {cfg.name!r}")
