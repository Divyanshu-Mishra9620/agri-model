from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.schema import AugmentationConfig, DataConfig

def build_train_transforms(data_cfg: DataConfig, aug_cfg: AugmentationConfig) -> A.Compose:
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
    size = data_cfg.image_size
    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.Normalize(mean=aug_cfg.mean, std=aug_cfg.std),
            ToTensorV2(),
        ]
    )
