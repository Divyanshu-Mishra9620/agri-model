"""PyTorch Dataset over a manifest (filepath, class_name) CSV/DataFrame."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CropDiseaseDataset(Dataset):
    """Reads images referenced by a manifest and applies an Albumentations
    transform.

    A missing/corrupt file at __getitem__ time (e.g. moved or deleted after
    the manifest was built) logs a warning and falls through to the next
    sample instead of crashing the whole epoch — `scripts/validate_dataset.py`
    is the primary defense against bad files entering the manifest at all;
    this is a last-resort safety net, not the main mechanism.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = (
            manifest.reset_index(drop=True)
            if isinstance(manifest, pd.DataFrame)
            else pd.read_csv(manifest)
        )
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_image(filepath: str) -> np.ndarray:
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        n = len(self)
        for attempt in range(n):
            row = self.df.iloc[(index + attempt) % n]
            try:
                image = self._load_image(row["filepath"])
            except FileNotFoundError as exc:
                logger.warning("%s — skipping to next sample.", exc)
                continue

            label = self.class_to_idx[row["class_name"]]
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image, label

        raise RuntimeError(
            "CropDiseaseDataset: every sample failed to load — the dataset "
            "appears to be broken. Run scripts/validate_dataset.py."
        )
