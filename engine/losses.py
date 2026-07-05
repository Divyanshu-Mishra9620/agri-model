"""Loss function factory: cross-entropy, weighted cross-entropy, focal loss,
label smoothing, and a focal+label-smoothing hybrid — selected via
`LossConfig.name`.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.schema import LossConfig

logger = logging.getLogger(__name__)


def _label_smoothed_nll(log_probs: torch.Tensor, targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Per-sample label-smoothed NLL, matching the formula behind
    `torch.nn.functional.cross_entropy(..., label_smoothing=epsilon)`:
    (1 - eps) * NLL(true class) + eps * mean(NLL over all classes).

    Implemented manually (rather than calling F.cross_entropy directly)
    because FocalLoss needs the per-sample loss *before* averaging, to scale
    each one by its focal weight.
    """
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return (1.0 - epsilon) * nll + epsilon * smooth


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional label smoothing:
    ``FL = (1 - p_t)^gamma * smoothed_NLL(p_t)``

    Downweights well-classified (typically majority-class) examples so
    gradient signal concentrates on hard/rare ones — a direct, per-example
    complement to the WeightedRandomSampler's per-class reweighting. The two
    operate on different axes: the sampler changes how often a class is
    *seen*, focal loss changes how much a *given* prediction contributes to
    the gradient.
    """

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
    """Build a [num_classes] tensor of inverse-frequency weights ordered by
    label index (not by name), normalized to mean 1.0 so the overall loss
    scale doesn't drift with num_classes."""
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
    """Build the configured loss function.

    `class_counts` + `class_to_idx` are only required for `loss.name='weighted_ce'`,
    to compute inverse-frequency class weights in label-index order.
    """
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
