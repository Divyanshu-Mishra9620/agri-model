"""CLI: build train/val/test manifests from the dataset (or its validated
clean manifest) using the imbalance-aware splitter in datasets/split.py.

Run from the repo root: `python scripts/prepare_splits.py`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.split import (
    build_class_to_idx,
    build_manifest_from_folder,
    create_splits,
    load_clean_manifest,
    save_splits,
)
from utils.config_loader import load_config
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _check_class_to_idx_stability(new_mapping: dict[str, int], splits_dir: Path, force: bool) -> None:
    """`class_to_idx.json` is the fixed source of truth a trained checkpoint's
    output indices are keyed against. If the dataset's class set has changed
    since the last split (a class added or removed) and this mapping were
    silently regenerated, every existing checkpoint's label mapping would be
    silently invalidated — predictions would still "work" (no crash) but
    point at the wrong disease names. Refuse to proceed unless the mapping
    is unchanged or the user explicitly passes --force.
    """
    existing_path = splits_dir / "class_to_idx.json"
    if not existing_path.exists():
        return

    existing_mapping = json.loads(existing_path.read_text(encoding="utf-8"))
    if existing_mapping == new_mapping:
        return

    added = sorted(set(new_mapping) - set(existing_mapping))
    removed = sorted(set(existing_mapping) - set(new_mapping))
    reindexed = sorted(
        name for name in set(new_mapping) & set(existing_mapping)
        if new_mapping[name] != existing_mapping[name]
    )
    message = (
        f"class_to_idx.json at {existing_path} would change "
        f"(added={added}, removed={removed}, reindexed={reindexed}). "
        f"Any existing checkpoint's class mapping would silently no longer "
        f"match the new indices."
    )
    if not force:
        raise RuntimeError(f"{message} Re-run with --force to proceed anyway (and retrain from scratch).")
    logger.warning("%s Proceeding because --force was passed.", message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test manifests.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument(
        "--force", action="store_true",
        help="Allow class_to_idx.json to change even if it invalidates existing checkpoints.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    setup_logging(cfg.log.log_dir, name="prepare_splits")

    clean_manifest_path = Path(cfg.data.clean_manifest)
    if clean_manifest_path.exists():
        logger.info("Using validated clean manifest: %s", clean_manifest_path)
        df = load_clean_manifest(clean_manifest_path)
    else:
        logger.warning(
            "No clean manifest found at %s — scanning data.root_dir directly. Run "
            "`python scripts/validate_dataset.py` first to filter out corrupted images "
            "before splitting.", clean_manifest_path,
        )
        df = build_manifest_from_folder(cfg.data.root_dir)

    class_to_idx = build_class_to_idx(df["class_name"].unique().tolist())
    _check_class_to_idx_stability(class_to_idx, Path(cfg.data.splits_dir), force=args.force)

    train_df, val_df, test_df, report = create_splits(df, cfg.data, seed=cfg.train.seed)
    save_splits(train_df, val_df, test_df, class_to_idx, report, cfg.data.splits_dir)

    logger.info("Done: %d classes -> %s", len(class_to_idx), cfg.data.splits_dir)


if __name__ == "__main__":
    main()
