"""Unit tests for the imbalance-aware split logic in datasets/split.py —
the highest-bug-risk part of this pipeline (stratification edge cases,
seed reproducibility, train/val/test leakage).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from configs.schema import DataConfig
from datasets.split import build_class_to_idx, create_splits


def _make_manifest(class_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for class_name, count in class_counts.items():
        for i in range(count):
            rows.append({"filepath": f"{class_name}/img_{i}.jpg", "class_name": class_name})
    return pd.DataFrame(rows)


@pytest.fixture
def cfg() -> DataConfig:
    return DataConfig(min_samples_for_split=10, min_samples_for_eval=3, val_split=0.1, test_split=0.1)


def test_tiny_class_is_train_only_and_excluded_from_eval(cfg):
    df = _make_manifest({"tiny": 2, "normal": 50})
    train_df, val_df, test_df, report = create_splits(df, cfg, seed=42)

    assert (train_df["class_name"] == "tiny").sum() == 2
    assert (val_df["class_name"] == "tiny").sum() == 0
    assert (test_df["class_name"] == "tiny").sum() == 0
    assert "tiny" in report.excluded_from_eval


def test_small_class_gets_forced_single_val_and_test_sample(cfg):
    df = _make_manifest({"small": 7, "normal": 50})
    train_df, val_df, test_df, report = create_splits(df, cfg, seed=42)

    assert (val_df["class_name"] == "small").sum() == 1
    assert (test_df["class_name"] == "small").sum() == 1
    assert (train_df["class_name"] == "small").sum() == 5
    assert "small" in report.small_class_forced_split


def test_normal_class_gets_proportional_split_with_at_least_one_each(cfg):
    df = _make_manifest({"normal": 100})
    train_df, val_df, test_df, report = create_splits(df, cfg, seed=42)

    assert len(val_df) >= 1
    assert len(test_df) >= 1
    assert len(train_df) + len(val_df) + len(test_df) == 100
    assert "normal" in report.normal_split


@pytest.mark.parametrize("n", [1, 2])
def test_pathological_min_samples_for_eval_never_empties_train(n):
    """Regression test: with a misconfigured (very low) min_samples_for_eval,
    a class landing in the small-class branch must still always get >=1
    training sample, never an empty train + non-empty val for a 1-image class."""
    cfg = DataConfig(min_samples_for_split=10, min_samples_for_eval=1, val_split=0.1, test_split=0.1)
    df = _make_manifest({"edge": n})

    train_df, val_df, test_df, _ = create_splits(df, cfg, seed=42)
    assert len(train_df) >= 1


def test_no_leakage_across_splits(cfg):
    df = _make_manifest({"a": 5, "b": 30, "c": 200})
    train_df, val_df, test_df, _ = create_splits(df, cfg, seed=42)

    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    assert len(train_paths) + len(val_paths) + len(test_paths) == len(df)


def test_split_is_reproducible_for_a_fixed_seed(cfg):
    df = _make_manifest({"a": 5, "b": 30, "c": 200})

    train1, val1, test1, _ = create_splits(df, cfg, seed=123)
    train2, val2, test2, _ = create_splits(df, cfg, seed=123)

    pd.testing.assert_frame_equal(train1.sort_values("filepath").reset_index(drop=True),
                                   train2.sort_values("filepath").reset_index(drop=True))
    pd.testing.assert_frame_equal(val1.sort_values("filepath").reset_index(drop=True),
                                   val2.sort_values("filepath").reset_index(drop=True))
    pd.testing.assert_frame_equal(test1.sort_values("filepath").reset_index(drop=True),
                                   test2.sort_values("filepath").reset_index(drop=True))


def test_different_seeds_can_produce_different_splits(cfg):
    df = _make_manifest({"a": 200})
    _, val1, _, _ = create_splits(df, cfg, seed=1)
    _, val2, _, _ = create_splits(df, cfg, seed=2)
    assert set(val1["filepath"]) != set(val2["filepath"])


def test_class_to_idx_is_alphabetically_sorted_and_stable():
    mapping = build_class_to_idx(["Zebra_rot", "Apple_scab", "Mango_blight"])
    assert mapping == {"Apple_scab": 0, "Mango_blight": 1, "Zebra_rot": 2}

    # Order of the input list must not matter — only the set of names does.
    shuffled_mapping = build_class_to_idx(["Mango_blight", "Zebra_rot", "Apple_scab"])
    assert mapping == shuffled_mapping
