"""Albumentations transform pipelines, built entirely from AugmentationConfig
so every probability/parameter is configurable via YAML — nothing hardcoded.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.schema import AugmentationConfig, DataConfig


def build_train_transforms(data_cfg: DataConfig, aug_cfg: AugmentationConfig) -> A.Compose:
    """Training-time augmentation pipeline.

    Probabilities are kept moderate by design: the disease signal itself is
    fine-grained leaf texture and color, so aggressive noise/dropout can
    destroy the cues the model needs to learn, not just add robustness.
    """
    size = data_cfg.image_size
    max_hole = max(2, int(size * aug_cfg.coarse_dropout_max_size_frac))

    return A.Compose(
        [
            A.RandomResizedCrop(size=(size, size), scale=(aug_cfg.random_resized_crop_scale_min, 1.0)),
            A.HorizontalFlip(p=aug_cfg.hflip_p),
            A.Rotate(limit=aug_cfg.rotate_limit, p=aug_cfg.rotate_p, border_mode=0),
            A.RandomBrightnessContrast(p=aug_cfg.brightness_contrast_p),
            A.ColorJitter(p=aug_cfg.color_jitter_p),
            A.CLAHE(p=aug_cfg.clahe_p),
            A.GaussNoise(p=aug_cfg.gauss_noise_p),
            A.CoarseDropout(
                num_holes_range=(1, max(1, aug_cfg.coarse_dropout_max_holes)),
                hole_height_range=(max(1, max_hole // 2), max_hole),
                hole_width_range=(max(1, max_hole // 2), max_hole),
                p=aug_cfg.coarse_dropout_p,
            ),
            A.Normalize(mean=aug_cfg.mean, std=aug_cfg.std),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(data_cfg: DataConfig, aug_cfg: AugmentationConfig) -> A.Compose:
    """Deterministic resize + normalize only, for validation/test/inference.

    Reused as-is by inference/predictor.py so serving preprocessing can
    never drift from what the model was validated against.
    """
    size = data_cfg.image_size
    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.Normalize(mean=aug_cfg.mean, std=aug_cfg.std),
            ToTensorV2(),
        ]
    )
