"""Training the edge model from photos that have no labels.

The cloud model (the teacher) already knows the diseases, so the small model
does not need labelled data to learn from it: it needs leaf photos, and the
teacher says what is on each one. Any folder of crop/leaf photos works - your
own phone or ESP32-CAM pictures, or a public dataset whose labels you ignore.

What this cannot do is learn about crops that never appear in the photos: the
small model only learns what the teacher says about the pictures it is shown.
And without labels there is no accuracy to measure, only how often the small
model agrees with the teacher on photos it was not trained on.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_LIST = "train_images.txt"
HOLDOUT_LIST = "holdout_images.txt"


def scan_images(dirs: Iterable[str | Path]) -> list[str]:
    paths: list[str] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"Image folder not found: {d}")
        paths.extend(str(p) for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return sorted(set(paths))


def split_holdout(paths: list[str], fraction: float, seed: int = 42) -> tuple[list[str], list[str]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n_hold = max(1, int(round(len(shuffled) * fraction)))
    return shuffled[n_hold:], shuffled[:n_hold]


def write_list(path: Path, items: list[str]) -> None:
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def read_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - it is written by edge.train_edge --images")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rgb(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class UnlabeledImages(Dataset):
    """Photos without labels. Yields (image, 0); the label is never used.

    An unreadable file is replaced by the next readable one, as in
    CropDiseaseDataset, so one bad photo does not stop an overnight run."""

    def __init__(self, paths: list[str], transform: Optional[Callable] = None):
        if not paths:
            raise ValueError("no images")
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        n = len(self.paths)
        for attempt in range(n):
            image = load_rgb(self.paths[(index + attempt) % n])
            if image is None:
                logger.warning("Could not read %s - skipping.", self.paths[(index + attempt) % n])
                continue
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image, 0
        raise RuntimeError("UnlabeledImages: no image in the set could be read")
