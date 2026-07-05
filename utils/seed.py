"""Reproducibility: seeding and deterministic-mode toggling."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG used across the pipeline (python, numpy, torch, CUDA).

    Args:
        seed: Seed value shared across all RNGs.
        deterministic: If True, forces deterministic cuDNN kernels and
            `torch.use_deterministic_algorithms` for bit-reproducible runs
            at a real speed cost. If False (default), cuDNN benchmark mode
            is enabled instead: faster, seeded, but not bit-exact across
            runs — the right trade-off for day-to-day experimentation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)
