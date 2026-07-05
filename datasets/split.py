"""Class-imbalance-aware train/val/test splitting.

With classes ranging from ~7 to ~60,000 images, a single global stratified
split (e.g. `sklearn.model_selection.train_test_split(stratify=...)`) either
raises on classes smaller than the number of splits, or silently starves
rare classes of validation/test coverage. This module splits *per class*
instead — which is what stratification means by construction — and applies
three explicit, always-safe rules by class size:

  n < min_samples_for_eval        -> all samples to train; excluded from val/test
  min_samples_for_eval <= n
      < min_samples_for_split     -> up to 1 sample each forced into val and
                                      test, remainder to train (never empties
                                      train, even for pathological configs)
  n >= min_samples_for_split      -> proportional split at val_split/test_split

Every class is handled predictably; nothing crashes and nothing is silently
dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from configs.schema import DataConfig

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["filepath", "class_name"]
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitReport:
    """Summary of how the split behaved — logged and saved alongside the
    manifests so an unusual dataset shape is always visible, never silent."""

    class_counts: dict[str, int] = field(default_factory=dict)
    excluded_from_eval: list[str] = field(default_factory=list)
    small_class_forced_split: list[str] = field(default_factory=list)
    normal_split: list[str] = field(default_factory=list)
    split_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_classes": len(self.class_counts),
            "class_counts": self.class_counts,
            "excluded_from_eval": self.excluded_from_eval,
            "small_class_forced_split": self.small_class_forced_split,
            "split_sizes": self.split_sizes,
        }


def build_manifest_from_folder(root_dir: str | Path) -> pd.DataFrame:
    """Scan an ImageFolder-style tree (root_dir/<ClassName>/<image>) into a
    flat manifest. Prefer `load_clean_manifest` (the output of
    scripts/validate_dataset.py) over this when it exists, so corrupted
    files never enter the split."""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root_dir}. Point data.root_dir at your "
            f"ImageFolder-style dataset (root_dir/<ClassName>/<image>.jpg)."
        )

    rows = []
    for class_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() in _VALID_EXTENSIONS:
                rows.append({"filepath": str(file_path), "class_name": class_dir.name})

    if not rows:
        raise ValueError(f"No images found under {root_dir}. Expected root_dir/<ClassName>/<image>.")

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    logger.info("Scanned %d images across %d classes from %s", len(df), df["class_name"].nunique(), root_dir)
    return df


def load_clean_manifest(path: str | Path) -> pd.DataFrame:
    """Load the manifest written by scripts/validate_dataset.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Clean manifest not found: {path}. Run `python scripts/validate_dataset.py` "
            f"first, or point data.clean_manifest at an existing one."
        )
    df = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing column(s): {missing}")
    return df[MANIFEST_COLUMNS]


def build_class_to_idx(class_names: list[str]) -> dict[str, int]:
    """Alphabetically sorted so the mapping is stable regardless of
    filesystem iteration order, which differs across OSes."""
    return {name: idx for idx, name in enumerate(sorted(set(class_names)))}


def create_splits(
    df: pd.DataFrame, cfg: DataConfig, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split `df` (columns: filepath, class_name) into train/val/test.

    A single `np.random.RandomState(seed)` is created once and threaded
    through every class's shuffle (rather than reseeding per class from
    `hash(class_name)` — Python's string hash is randomized per-process by
    `PYTHONHASHSEED` and setting that env var at runtime does not affect the
    already-running interpreter, so it would silently break reproducibility).
    Classes are iterated in a fixed sorted order so the whole split is fully
    deterministic for a given seed.
    """
    report = SplitReport()
    train_parts, val_parts, test_parts = [], [], []
    rng = np.random.RandomState(seed)

    for class_name, group in df.groupby("class_name", sort=True):
        n = len(group)
        report.class_counts[class_name] = n
        shuffled = group.sample(frac=1.0, random_state=rng).reset_index(drop=True)

        if n < cfg.min_samples_for_eval:
            train_parts.append(shuffled)
            report.excluded_from_eval.append(class_name)
            continue

        if n < cfg.min_samples_for_split:
            # Guarantee train always gets >=1 sample, even for pathological
            # configs (e.g. min_samples_for_eval set very low): only carve
            # out a test sample if n>=2, and only a val sample if one still
            # remains for train afterwards.
            test_n = 1 if n >= 2 else 0
            val_n = 1 if (n - test_n) >= 2 else 0
            test_part = shuffled.iloc[:test_n]
            val_part = shuffled.iloc[test_n : test_n + val_n]
            train_part = shuffled.iloc[test_n + val_n :]
            report.small_class_forced_split.append(class_name)
        else:
            test_n = max(1, round(n * cfg.test_split))
            val_n = max(1, round(n * cfg.val_split))
            test_part = shuffled.iloc[:test_n]
            val_part = shuffled.iloc[test_n : test_n + val_n]
            train_part = shuffled.iloc[test_n + val_n :]
            report.normal_split.append(class_name)

        train_parts.append(train_part)
        if len(val_part):
            val_parts.append(val_part)
        if len(test_part):
            test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=MANIFEST_COLUMNS)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=MANIFEST_COLUMNS)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=MANIFEST_COLUMNS)

    report.split_sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}

    if report.excluded_from_eval:
        logger.warning(
            "%d class(es) have fewer than data.min_samples_for_eval=%d images and are "
            "TRAIN-ONLY (no val/test coverage): %s",
            len(report.excluded_from_eval), cfg.min_samples_for_eval, report.excluded_from_eval,
        )
    if report.small_class_forced_split:
        logger.warning(
            "%d class(es) have fewer than data.min_samples_for_split=%d images; forced to "
            "at most 1 val + 1 test sample each (their val/test metrics will be noisy): %s",
            len(report.small_class_forced_split), cfg.min_samples_for_split, report.small_class_forced_split,
        )

    logger.info("Split sizes: %s", report.split_sizes)
    return train_df, val_df, test_df, report


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_to_idx: dict[str, int],
    report: SplitReport,
    splits_dir: str | Path,
) -> None:
    """Persist the splits + class mapping + report for reproducibility.

    `class_to_idx.json` is also what predict.py / inference.predictor use to
    map model output indices back to human-readable disease names.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "manifest_train.csv", index=False)
    val_df.to_csv(splits_dir / "manifest_val.csv", index=False)
    test_df.to_csv(splits_dir / "manifest_test.csv", index=False)

    with (splits_dir / "class_to_idx.json").open("w", encoding="utf-8") as fh:
        json.dump(class_to_idx, fh, indent=2, ensure_ascii=False)

    with (splits_dir / "split_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)

    logger.info("Wrote splits + class_to_idx.json + split_report.json to %s", splits_dir)
