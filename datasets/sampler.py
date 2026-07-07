from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)

def compute_class_weights(class_counts: dict[str, int], power: float = 1.0) -> dict[str, float]:
    return {name: (1.0 / count) ** power for name, count in class_counts.items()}

def build_weighted_sampler(train_df: pd.DataFrame, max_weight_ratio: float = 20.0) -> WeightedRandomSampler:
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
