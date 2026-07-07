from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from configs.schema import AugmentationConfig, DataConfig
from datasets.transforms import build_eval_transforms, build_train_transforms

@pytest.fixture
def synthetic_image() -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)

@pytest.mark.parametrize("image_size", [128, 224])
def test_train_transform_output_shape_and_dtype(synthetic_image, image_size):
    data_cfg = DataConfig(image_size=image_size)
    aug_cfg = AugmentationConfig()
    transform = build_train_transforms(data_cfg, aug_cfg)

    output = transform(image=synthetic_image)["image"]

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, image_size, image_size)
    assert output.dtype == torch.float32

def test_eval_transform_output_shape_and_dtype(synthetic_image):
    data_cfg = DataConfig(image_size=224)
    aug_cfg = AugmentationConfig()
    transform = build_eval_transforms(data_cfg, aug_cfg)

    output = transform(image=synthetic_image)["image"]

    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32

def test_eval_transform_is_deterministic(synthetic_image):
    data_cfg = DataConfig(image_size=224)
    aug_cfg = AugmentationConfig()
    transform = build_eval_transforms(data_cfg, aug_cfg)

    output_1 = transform(image=synthetic_image)["image"]
    output_2 = transform(image=synthetic_image)["image"]

    torch.testing.assert_close(output_1, output_2)

def test_eval_transform_normalizes_to_roughly_zero_mean(synthetic_image):
    data_cfg = DataConfig(image_size=224)
    aug_cfg = AugmentationConfig()
    transform = build_eval_transforms(data_cfg, aug_cfg)

    output = transform(image=synthetic_image)["image"]

    assert output.mean().abs().item() < 3.0 

def test_train_transform_produces_varied_output_across_calls(synthetic_image):
    data_cfg = DataConfig(image_size=224)
    aug_cfg = AugmentationConfig()
    transform = build_train_transforms(data_cfg, aug_cfg)

    torch.manual_seed(0)
    output_1 = transform(image=synthetic_image)["image"]
    torch.manual_seed(1)
    output_2 = transform(image=synthetic_image)["image"]

    assert not torch.allclose(output_1, output_2)
