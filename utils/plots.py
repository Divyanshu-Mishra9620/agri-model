"""Matplotlib visualization helpers.

Uses the non-interactive Agg backend so these work headless (training
server / CI with no display).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_curve(
    values_by_split: dict[str, Sequence[float]],
    *,
    title: str,
    ylabel: str,
    path: str | Path,
) -> None:
    """Generic train/val curve plot (e.g. loss or accuracy over epochs)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split_name, values in values_by_split.items():
        ax.plot(range(1, len(values) + 1), values, label=split_name, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: str | Path,
    *,
    max_labeled_classes: int = 40,
) -> None:
    """Render a confusion-matrix heatmap.

    With up to ~180 classes, a fully axis-labeled heatmap is unreadable, so
    tick labels are only drawn when `len(class_names) <= max_labeled_classes`.
    Above that the image still renders (the diagonal/pattern is visually
    informative) — pair it with `plot_worst_classes` for readable per-class
    detail.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({len(class_names)} classes)")

    if len(class_names) <= max_labeled_classes:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    _save(fig, path)


def plot_worst_classes(per_class_accuracy: dict[str, float], path: str | Path, *, top_n: int = 20) -> None:
    """Horizontal bar chart of the `top_n` lowest-accuracy classes."""
    worst = sorted(per_class_accuracy.items(), key=lambda kv: kv[1])[:top_n]
    names = [name for name, _ in worst]
    values = [value for _, value in worst]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(names))))
    ax.barh(names, values, color="#c0392b")
    ax.set_xlabel("Accuracy")
    ax.set_title(f"{len(names)} Worst-Performing Classes")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    _save(fig, path)


def plot_class_distribution(class_counts: dict[str, int], path: str | Path) -> None:
    """Ascending-sorted bar chart of images-per-class (log scale), making the
    long tail of rare classes visually obvious."""
    items = sorted(class_counts.items(), key=lambda kv: kv[1])
    names = [name for name, _ in items]
    counts = [count for _, count in items]

    fig, ax = plt.subplots(figsize=(max(8, 0.12 * len(names)), 5))
    ax.bar(range(len(names)), counts, color="#2e7d32")
    ax.set_yscale("log")
    ax.set_ylabel("Image count (log scale)")
    ax.set_title("Class Distribution (ascending)")
    ax.set_xticks([])
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path)
