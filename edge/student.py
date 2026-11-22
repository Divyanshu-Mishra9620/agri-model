from __future__ import annotations

import logging

import timm
import torch.nn as nn

logger = logging.getLogger(__name__)

# MobileNetV2 at width 0.5 is the default because it is built only from ops the
# gateway's int8 engine implements (conv, depthwise conv, ReLU6, add, average
# pool, fully connected). MobileNetV3 / EfficientNet use hard-swish, SiLU and
# squeeze-excite, which the engine does not have.
DEFAULT_STUDENT = "mobilenetv2_050"
DEFAULT_IMAGE_SIZE = 160

SUPPORTED_STUDENTS = (
    "mobilenetv2_035",
    "mobilenetv2_050",
    "mobilenetv2_075",
    "mobilenetv2_100",
)


def build_student(
    name: str = DEFAULT_STUDENT,
    num_classes: int = -1,
    pretrained: bool = True,
    drop_rate: float = 0.1,
) -> nn.Module:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if name not in SUPPORTED_STUDENTS:
        raise ValueError(
            f"Student {name!r} is not supported by the gateway's int8 engine. "
            f"Pick one of: {', '.join(SUPPORTED_STUDENTS)}"
        )

    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built student %s (pretrained=%s) with %d classes - %.2fM parameters (~%.2f MB as int8).",
        name, pretrained, num_classes, n_params / 1e6, n_params / 1e6,
    )
    return model
