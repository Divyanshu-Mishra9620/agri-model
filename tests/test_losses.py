from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from configs.schema import LossConfig
from engine.losses import FocalLoss, build_loss

N_CLASSES = 6
BATCH_SIZE = 8

@pytest.fixture
def logits_and_targets():
    torch.manual_seed(0)
    logits = torch.randn(BATCH_SIZE, N_CLASSES, requires_grad=True)
    targets = torch.randint(0, N_CLASSES, (BATCH_SIZE,))
    return logits, targets

@pytest.mark.parametrize("loss_name", ["ce", "label_smoothing", "focal", "focal_ls"])
def test_loss_produces_finite_scalar_and_is_differentiable(loss_name, logits_and_targets):
    logits, targets = logits_and_targets
    loss_fn = build_loss(LossConfig(name=loss_name))

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

def test_weighted_ce_requires_class_counts_and_class_to_idx():
    with pytest.raises(ValueError, match="weighted_ce"):
        build_loss(LossConfig(name="weighted_ce"))

def test_weighted_ce_builds_with_class_counts(logits_and_targets):
    logits, targets = logits_and_targets
    class_to_idx = {f"class_{i}": i for i in range(N_CLASSES)}
    class_counts = {name: (i + 1) * 10 for i, name in enumerate(class_to_idx)}

    loss_fn = build_loss(LossConfig(name="weighted_ce"), class_counts=class_counts, class_to_idx=class_to_idx)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)

def test_unknown_loss_name_raises():
    with pytest.raises(ValueError, match="Unknown loss.name"):
        build_loss(LossConfig(name="not_a_real_loss"))

def test_focal_loss_with_gamma_zero_matches_plain_cross_entropy(logits_and_targets):
    logits, targets = logits_and_targets
    focal = FocalLoss(gamma=0.0, label_smoothing=0.0)
    plain_ce = nn.CrossEntropyLoss()

    torch.testing.assert_close(focal(logits, targets), plain_ce(logits, targets))

def test_focal_loss_downweights_easy_examples():
    torch.manual_seed(0)
    easy_logits = torch.tensor([[4.0, -1.0, -1.0]]) 
    target = torch.tensor([0])

    focal_loss_value = FocalLoss(gamma=2.0).forward(easy_logits, target)
    ce_loss_value = nn.CrossEntropyLoss()(easy_logits, target)

    assert focal_loss_value.item() < ce_loss_value.item()
