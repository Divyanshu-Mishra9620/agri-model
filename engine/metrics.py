"""Metric accumulation and computation for classification: top-1/3 accuracy,
precision/recall/F1 (macro & weighted), per-class accuracy, and the
confusion matrix.

Shared by Trainer's validation loop and evaluate.py so training-time and
standalone evaluation always compute metrics exactly the same way — one
implementation, not two that could drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    loss: float
    top1_accuracy: float
    top3_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    macro_f1_eval_subset: float
    """Macro-F1 restricted to classes with >= `min_eval_samples` support in
    this split. This is the metric `train.early_stopping_metric` should
    name: unrestricted macro-F1 can swing by 1/num_classes on a single
    borderline prediction for a 1-sample class, independent of real model
    quality."""
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    confusion: np.ndarray | None = None

    def to_log_dict(self) -> dict[str, float]:
        """Flat dict of scalar metrics only, for CSV/TensorBoard logging."""
        return {
            "loss": self.loss,
            "top1_accuracy": self.top1_accuracy,
            "top3_accuracy": self.top3_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "macro_f1_eval_subset": self.macro_f1_eval_subset,
        }


class MetricTracker:
    """Accumulates logits/targets/loss across an epoch, then computes a full
    `EpochMetrics` on demand via `compute()`."""

    def __init__(self, idx_to_class: dict[int, str], min_eval_samples: int = 2):
        self.idx_to_class = idx_to_class
        self.min_eval_samples = min_eval_samples
        self.reset()

    def reset(self) -> None:
        self._logits: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._loss_sum = 0.0
        self._loss_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        self._logits.append(logits.detach().float().cpu())
        self._targets.append(targets.detach().cpu())
        self._loss_sum += loss * targets.size(0)
        self._loss_count += targets.size(0)

    @torch.no_grad()
    def compute(self) -> EpochMetrics:
        if not self._logits:
            raise RuntimeError("MetricTracker.compute() called with no accumulated batches.")

        logits = torch.cat(self._logits)
        targets = torch.cat(self._targets)
        num_classes = logits.size(1)
        labels = list(range(num_classes))

        top1, top3 = _topk_accuracy(logits, targets, ks=(1, 3))

        preds = logits.argmax(dim=1).numpy()
        targets_np = targets.numpy()

        precision, recall, f1, support = precision_recall_fscore_support(
            targets_np, preds, labels=labels, average=None, zero_division=0
        )
        f1_weighted = float(np.average(f1, weights=support)) if support.sum() > 0 else 0.0

        eval_subset = support >= self.min_eval_samples
        macro_f1_eval_subset = float(np.mean(f1[eval_subset])) if eval_subset.any() else 0.0
        if not eval_subset.all():
            n_excluded = int((~eval_subset).sum())
            logger.debug(
                "%d/%d classes have < %d samples in this split and are excluded "
                "from macro_f1_eval_subset.", n_excluded, num_classes, self.min_eval_samples,
            )

        confusion = confusion_matrix(targets_np, preds, labels=labels)

        return EpochMetrics(
            loss=self._loss_sum / max(1, self._loss_count),
            top1_accuracy=top1,
            top3_accuracy=top3,
            precision_macro=float(np.mean(precision)),
            recall_macro=float(np.mean(recall)),
            f1_macro=float(np.mean(f1)),
            f1_weighted=f1_weighted,
            macro_f1_eval_subset=macro_f1_eval_subset,
            per_class_accuracy=_per_class_accuracy(confusion, self.idx_to_class),
            confusion=confusion,
        )


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...]) -> tuple[float, ...]:
    max_k = min(max(ks), logits.size(1))
    _, top_preds = logits.topk(max_k, dim=1)
    correct = top_preds.eq(targets.unsqueeze(1))
    return tuple(correct[:, : min(k, max_k)].any(dim=1).float().mean().item() for k in ks)


def _per_class_accuracy(confusion: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, float]:
    accuracies = {}
    for idx in range(confusion.shape[0]):
        support = confusion[idx].sum()
        accuracies[idx_to_class[idx]] = float(confusion[idx, idx] / support) if support > 0 else float("nan")
    return accuracies
