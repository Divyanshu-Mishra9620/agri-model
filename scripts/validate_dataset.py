from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image

logger = logging.getLogger(__name__)

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass
class ImageCheckResult:
    filepath: str
    class_name: str
    is_valid: bool
    error: Optional[str] = None
    phash: Optional[str] = None

def _check_one(filepath: str, class_name: str) -> ImageCheckResult:
    import imagehash  

    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            phash = str(imagehash.phash(img))
        return ImageCheckResult(filepath=filepath, class_name=class_name, is_valid=True, phash=phash)
    except Exception as exc:  
        return ImageCheckResult(filepath=filepath, class_name=class_name, is_valid=False, error=str(exc))

def _check_image_chunk(items: list[tuple[str, str]]) -> list[ImageCheckResult]:
    return [_check_one(filepath, class_name) for filepath, class_name in items]

def _make_chunks(files: list[tuple[str, str]], chunk_size: int) -> list[list[tuple[str, str]]]:
    return [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

def _iter_dataset_files(root_dir: Path):
    for class_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() in _VALID_EXTENSIONS:
                yield str(file_path), class_dir.name

def validate_dataset(
    root_dir: Path,
    output_dir: Path,
    *,
    max_workers: int = 8,
    rare_class_threshold: int = 20,
    heartbeat_seconds: float = 30.0,
    chunk_size: Optional[int] = None,
    target_chunk_count: int = 400,
) -> dict:
    import pandas as pd
    from tqdm import tqdm

    files = list(_iter_dataset_files(root_dir))
    if not files:
        raise ValueError(f"No images found under {root_dir}")

    if chunk_size is None:
        chunk_size = max(1, len(files) // target_chunk_count)
    chunks = _make_chunks(files, chunk_size)

    logger.info(
        "Validating %d images across %d classes with %d worker processes "
        "(%d chunks of ~%d images each)...",
        len(files), len({c for _, c in files}), max_workers, len(chunks), chunk_size,
    )

    results: list[ImageCheckResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        submit_start = time.monotonic()
        futures = [executor.submit(_check_image_chunk, chunk) for chunk in chunks]
        logger.info("Submitted %d chunk(s) in %.1fs — waiting on worker processes...", len(futures), time.monotonic() - submit_start)
        last_heartbeat = time.monotonic()
        progress = tqdm(total=len(files), desc="Validating images")
        for future in as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)
            progress.update(len(chunk_results))
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_seconds:
                logger.info("Heartbeat: %d/%d images processed (%.0fs elapsed).", len(results), len(files), now - submit_start)
                last_heartbeat = now
        progress.close()

    valid = [r for r in results if r.is_valid]
    corrupted = [r for r in results if not r.is_valid]

    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for r in valid:
        hash_to_paths[r.phash].append(r.filepath)
    exact_duplicate_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    class_counts: dict[str, int] = defaultdict(int)
    for r in valid:
        class_counts[r.class_name] += 1
    rare_classes = {name: count for name, count in class_counts.items() if count < rare_class_threshold}

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"filepath": r.filepath, "class_name": r.class_name} for r in valid]).to_csv(
        output_dir / "clean_manifest.csv", index=False
    )

    if corrupted:
        pd.DataFrame(
            [{"filepath": r.filepath, "class_name": r.class_name, "error": r.error} for r in corrupted]
        ).to_csv(output_dir / "corrupted_images.csv", index=False)

    if exact_duplicate_groups:
        dup_rows = [
            {"phash": h, "count": len(paths), "filepaths": "; ".join(paths)}
            for h, paths in sorted(exact_duplicate_groups.items(), key=lambda kv: -len(kv[1]))
        ]
        pd.DataFrame(dup_rows).to_csv(output_dir / "duplicate_report.csv", index=False)

    pd.DataFrame(
        sorted(class_counts.items(), key=lambda kv: kv[1]), columns=["class_name", "count"]
    ).to_csv(output_dir / "class_distribution.csv", index=False)

    from utils.plots import plot_class_distribution  

    plot_class_distribution(dict(class_counts), output_dir / "class_distribution.png")

    report = {
        "total_images_scanned": len(files),
        "valid_images": len(valid),
        "corrupted_images": len(corrupted),
        "num_classes": len(class_counts),
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "images_in_duplicate_groups": sum(len(p) for p in exact_duplicate_groups.values()),
        "rare_classes_below_threshold": rare_classes,
        "rare_class_threshold": rare_class_threshold,
        "class_counts": dict(class_counts),
        "note": (
            "Duplicate detection covers exact perceptual-hash matches only; "
            "see module docstring for why near-duplicate detection is out of scope."
        ),
    }
    with (output_dir / "dataset_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _write_markdown_summary(report, output_dir / "dataset_report.md")

    logger.info(
        "Validation complete: %d valid, %d corrupted, %d classes, %d exact-duplicate groups, %d rare classes.",
        len(valid), len(corrupted), len(class_counts), len(exact_duplicate_groups), len(rare_classes),
    )
    if corrupted:
        logger.warning(
            "%d corrupted/unreadable image(s) excluded from clean_manifest.csv (NOT deleted from "
            "disk) — see corrupted_images.csv.", len(corrupted),
        )
    if rare_classes:
        logger.warning(
            "%d class(es) have fewer than %d images: %s",
            len(rare_classes), rare_class_threshold, list(rare_classes),
        )

    return report

def _write_markdown_summary(report: dict, path: Path) -> None:
    lines = [
        "# Dataset Validation Report",
        "",
        f"- Total images scanned: **{report['total_images_scanned']}**",
        f"- Valid images: **{report['valid_images']}**",
        f"- Corrupted/unreadable images: **{report['corrupted_images']}** "
        f"(see corrupted_images.csv — files were NOT deleted)",
        f"- Classes: **{report['num_classes']}**",
        f"- Exact-duplicate groups: **{report['exact_duplicate_groups']}** "
        f"({report['images_in_duplicate_groups']} images total — see duplicate_report.csv)",
        f"- Classes below rarity threshold ({report['rare_class_threshold']} images): "
        f"**{len(report['rare_classes_below_threshold'])}**",
        "",
        "## Rare classes",
        "",
    ]
    for name, count in sorted(report["rare_classes_below_threshold"].items(), key=lambda kv: kv[1]):
        lines.append(f"- `{name}`: {count} images")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a crop-disease image dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--rare-class-threshold", type=int, default=20)
    parser.add_argument(
        "--heartbeat-seconds", type=float, default=30.0,
        help="Log a plain progress line at least this often, so a slow-but-alive run is never "
             "mistaken for a hang (tqdm's own bar can look static in some terminals/log redirects).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Images per worker task. Default: auto-computed to target ~400 total chunks "
             "regardless of dataset size — one task per image is what makes submission slow "
             "at 300k+ images (measured: ~1ms/call, minutes of pure overhead).",
    )
    args = parser.parse_args()

    from utils.config_loader import load_config 
    from utils.logging_utils import setup_logging 

    cfg = load_config(args.config)
    setup_logging(cfg.log.log_dir, name="validate_dataset")

    output_dir = Path(cfg.data.clean_manifest).parent
    validate_dataset(
        root_dir=Path(cfg.data.root_dir),
        output_dir=output_dir,
        max_workers=args.max_workers,
        rare_class_threshold=args.rare_class_threshold,
        heartbeat_seconds=args.heartbeat_seconds,
        chunk_size=args.chunk_size,
    )

if __name__ == "__main__":
    main()
