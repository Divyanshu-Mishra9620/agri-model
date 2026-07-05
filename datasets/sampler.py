"""WeightedRandomSampler construction for class-imbalanced training.

Only meaningful when `data.use_weighted_sampler=True`, and deliberately
discouraged from being combined with `loss.name='weighted_ce'` — both
correct for the same imbalance, and stacking them can overcorrect rare-class
gradients into instability. See configs/schema.py::LossConfig and
utils/config_loader.py::_validate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


def compute_class_weights(class_counts: dict[str, int], power: float = 1.0) -> dict[str, float]:
    """Inverse-frequency class weights: weight = (1 / count) ** power."""
    return {name: (1.0 / count) ** power for name, count in class_counts.items()}


def build_weighted_sampler(train_df: pd.DataFrame, max_weight_ratio: float = 20.0) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler over `train_df` (column: class_name)
    using inverse-class-frequency per-sample weights, capped at
    `max_weight_ratio` times the MEDIAN class weight.

    Why the cap: with class counts spanning ~7 to ~60,000 images (roughly an
    8500:1 ratio), raw inverse-frequency weights would make the rarest
    class's samples so overwhelmingly likely to be drawn that a large slice
    of every epoch becomes the same handful of images on repeat — badly
    overfitting that class while starving gradient updates for everything
    else. Clipping the ratio still strongly favors rare classes without
    letting any single one dominate the epoch.
    """
    class_counts = train_df["class_name"].value_counts().to_dict()
    raw_weights = compute_class_weights(class_counts)

    median_weight = float(np.median(list(raw_weights.values())))
    cap = median_weight * max_weight_ratio
    clipped_weights = {name: min(weight, cap) for name, weight in raw_weights.items()}

    n_clipped = sum(1 for w in raw_weights.values() if w > cap)
    if n_clipped:
        logger.info(
            "WeightedRandomSampler: clipped %d/%d class weight(s) to %.1fx "
            "the median class weight (data.max_sample_weight_ratio=%.1f).",
            n_clipped, len(raw_weights), max_weight_ratio, max_weight_ratio,
        )

    sample_weights = train_df["class_name"].map(clipped_weights).to_numpy(dtype=np.float64)
    n = len(train_df)
    total_weight = sample_weights.sum()
    expected_draws = {name: n * (clipped_weights[name] / total_weight) * count for name, count in class_counts.items()}
    smallest_class = min(class_counts, key=class_counts.get)
    largest_class = max(class_counts, key=class_counts.get)
    logger.info(
        "WeightedRandomSampler expected draws/epoch — smallest class %r (%d images): "
        "~%.0f draws; largest class %r (%d images): ~%.0f draws.",
        smallest_class, class_counts[smallest_class], expected_draws[smallest_class],
        largest_class, class_counts[largest_class], expected_draws[largest_class],
    )

    return WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=n, replacement=True)
