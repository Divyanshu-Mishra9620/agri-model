"""Benchmark DataLoader throughput across a range of num_workers values, so
`data.num_workers` can be chosen empirically instead of guessed — especially
useful on Windows, where multiprocessing (spawn-based, not fork) overhead
differs meaningfully from Linux.

Run from the repo root: `python scripts/benchmark_dataloader.py`
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from configs.schema import Config
from datasets.disease_dataset import CropDiseaseDataset
from datasets.split import build_class_to_idx, load_clean_manifest
from datasets.transforms import build_train_transforms
from utils.config_loader import load_config
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def benchmark(cfg: Config, manifest_path: Path, num_workers_options: list[int], num_batches: int) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run scripts/validate_dataset.py and "
            f"scripts/prepare_splits.py first."
        )

    df = load_clean_manifest(manifest_path)
    class_to_idx = build_class_to_idx(df["class_name"].unique().tolist())
    transform = build_train_transforms(cfg.data, cfg.augmentation)
    dataset = CropDiseaseDataset(df, class_to_idx, transform=transform)

    for num_workers in num_workers_options:
        loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.persistent_workers and num_workers > 0,
            prefetch_factor=cfg.data.prefetch_factor if num_workers > 0 else None,
        )

        try:
            it = iter(loader)
            next(it)  # warm-up: excludes one-time worker startup cost from the timed region

            start = time.perf_counter()
            n_images = 0
            for _ in range(num_batches):
                images, _ = next(it)
                n_images += images.size(0)
            elapsed = time.perf_counter() - start

            logger.info(
                "num_workers=%2d | %6.1f img/s | %.3fs for %d batches (batch_size=%d)",
                num_workers, n_images / elapsed, elapsed, num_batches, cfg.data.batch_size,
            )
        except StopIteration:
            logger.warning(
                "num_workers=%d: manifest has fewer than %d batches (+1 warm-up) at "
                "batch_size=%d — reduce --num-batches or point at a larger manifest.",
                num_workers, num_batches, cfg.data.batch_size,
            )
        finally:
            del loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader throughput.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default=None, help="Defaults to data.splits_dir/manifest_train.csv")
    parser.add_argument("--num-workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--num-batches", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.log.log_dir, name="benchmark_dataloader")

    manifest_path = Path(args.manifest) if args.manifest else Path(cfg.data.splits_dir) / "manifest_train.csv"
    benchmark(cfg, manifest_path, args.num_workers, args.num_batches)


if __name__ == "__main__":
    # Required on Windows: num_workers > 0 uses spawn-based multiprocessing,
    # which re-imports this module in each worker.
    main()
