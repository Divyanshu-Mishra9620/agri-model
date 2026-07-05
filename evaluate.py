"""Standalone evaluation on the held-out test set: computes the full metric
suite (top-1/3 accuracy, precision/recall/F1, confusion matrix, per-class
accuracy) using the exact same engine.metrics.MetricTracker training-time
validation uses, plus a many/medium/few-shot bucket breakdown — the standard
long-tail-recognition reporting convention, so a single macro number can't
hide that rare classes are systematically worse served.

Usage:
    python evaluate.py --checkpoint checkpoints/convnextv2_tiny_baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.disease_dataset import CropDiseaseDataset
from datasets.transforms import build_eval_transforms
from engine.metrics import MetricTracker
from inference.predictor import load_model_from_checkpoint
from utils.device import get_device, prepare_model, to_channels_last
from utils.logging_utils import setup_logging
from utils.plots import plot_confusion_matrix, plot_worst_classes

logger = logging.getLogger(__name__)


def _bucket_report(class_counts: dict[str, int], per_class_accuracy: dict[str, float]) -> dict:
    """Group per-class accuracy by how much TEST data each class actually
    had (not train count — a class can have thousands of train images but
    only 1-2 test images near the min_samples_for_split boundary). Standard
    long-tail-recognition thresholds: many->=100, medium 20-99, few <20.
    """
    buckets = {"many_shot (>=100 imgs)": [], "medium_shot (20-99 imgs)": [], "few_shot (<20 imgs)": []}
    for name, count in class_counts.items():
        acc = per_class_accuracy.get(name)
        if acc is None or math.isnan(acc):  # NaN: class had 0 test support, nothing to bucket
            continue
        bucket = "many_shot (>=100 imgs)" if count >= 100 else "medium_shot (20-99 imgs)" if count >= 20 else "few_shot (<20 imgs)"
        buckets[bucket].append(acc)

    return {
        bucket: {"num_classes": len(accs), "mean_accuracy": (sum(accs) / len(accs)) if accs else None}
        for bucket, accs in buckets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the held-out test set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None, help="Defaults to <data.splits_dir>/manifest_test.csv")
    parser.add_argument("--output-dir", default=None, help="Defaults to outputs/<experiment_name>/evaluate")
    args = parser.parse_args()

    model, cfg, class_to_idx = load_model_from_checkpoint(args.checkpoint)
    setup_logging(cfg.log.log_dir, name="evaluate")

    device = get_device()
    model = prepare_model(model, device, channels_last=cfg.train.channels_last, compile_model=False)

    manifest_path = Path(args.manifest) if args.manifest else Path(cfg.data.splits_dir) / "manifest_test.csv"
    test_df = pd.read_csv(manifest_path)
    dataset = CropDiseaseDataset(test_df, class_to_idx, transform=build_eval_transforms(cfg.data, cfg.augmentation))
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    tracker = MetricTracker(idx_to_class, min_eval_samples=cfg.train.early_stopping_min_eval_samples)

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = to_channels_last(images.to(device), enabled=cfg.train.channels_last)
            targets = targets.to(device)
            logits = model(images)
            # Plain CE regardless of training loss: keeps the reported test
            # loss on one fixed, comparable scale across runs trained with
            # different loss.name choices (focal/weighted_ce aren't even
            # numerically comparable to each other).
            loss = F.cross_entropy(logits, targets)
            tracker.update(logits, targets, loss.item())

    metrics = tracker.compute()

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.log.output_dir) / cfg.log.experiment_name / "evaluate"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts = test_df["class_name"].value_counts().to_dict()
    bucket_report = _bucket_report(class_counts, metrics.per_class_accuracy)

    summary = {
        **metrics.to_log_dict(),
        "num_test_images": len(test_df),
        "num_classes": len(class_to_idx),
        "bucket_report": bucket_report,
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    per_class_df = pd.DataFrame(metrics.per_class_accuracy.items(), columns=["class_name", "accuracy"])
    per_class_df["is_nan"] = per_class_df["accuracy"].isna()
    per_class_df.sort_values(["is_nan", "accuracy"]).drop(columns="is_nan").to_csv(
        output_dir / "per_class_accuracy.csv", index=False
    )

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    plot_confusion_matrix(metrics.confusion, class_names, output_dir / "confusion_matrix.png")
    plot_worst_classes(
        {k: v for k, v in metrics.per_class_accuracy.items() if not math.isnan(v)},
        output_dir / "worst_classes.png",
    )

    logger.info(
        "Test results: top1=%.4f top3=%.4f macro_f1=%.4f weighted_f1=%.4f",
        metrics.top1_accuracy, metrics.top3_accuracy, metrics.f1_macro, metrics.f1_weighted,
    )
    logger.info("Bucket report: %s", json.dumps(bucket_report, indent=2))
    logger.info("Full report written to %s", output_dir)


if __name__ == "__main__":
    main()
