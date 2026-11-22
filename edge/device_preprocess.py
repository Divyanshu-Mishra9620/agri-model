"""The gateway's preprocessing, reproduced exactly in numpy.

On the gateway (kn_edge.cpp) a decoded photo becomes the model input in two
steps, and this file does the same two steps so that accuracy measured here is
accuracy on what the device actually feeds the network:

1. Box resize to the model's square input (the photo is squashed, not cropped,
   matching the A.Resize used in training). Output pixel (ox, oy) is the
   rounded mean of the source pixels in
       x in [floor(ox*W/Wo), floor((ox+1)*W/Wo)),  y likewise.
   When the source is larger than the output - the only case the gateway
   handles - these boxes tile the photo exactly, so the device can accumulate
   them MCU by MCU while decoding, without ever holding the full image.
2. Per-channel lookup table from 0..255 to the int8 input of the network
   (normalisation and quantisation folded together; the table itself is
   shipped inside the generated kn_edge_model.cpp).
"""
from __future__ import annotations

import cv2
import numpy as np


def box_bounds(src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
    o = np.arange(dst, dtype=np.int64)
    lo = (o * src) // dst
    hi = np.maximum(lo + 1, ((o + 1) * src) // dst)
    return lo, hi


def box_resize(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """(H, W, 3) uint8 -> (out_h, out_w, 3) uint8, integer-exact box filter."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an (H, W, 3) uint8 RGB image")
    h, w = rgb.shape[:2]
    integral = np.zeros((h + 1, w + 1, 3), dtype=np.int64)
    integral[1:, 1:] = rgb.astype(np.int64).cumsum(0).cumsum(1)
    x0, x1 = box_bounds(w, out_w)
    y0, y1 = box_bounds(h, out_h)
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    s = integral[Y1, X1] - integral[Y0, X1] - integral[Y1, X0] + integral[Y0, X0]
    cnt = ((Y1 - Y0) * (X1 - X0))[..., None]
    return ((s + cnt // 2) // cnt).astype(np.uint8)


def to_model_input(rgb_small: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W, 3) int8 through the per-channel table."""
    return np.stack([lut[c][rgb_small[..., c]] for c in range(3)], axis=-1).astype(np.int8)


def load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def device_input(path_or_rgb, size: int, lut: np.ndarray) -> np.ndarray:
    rgb = load_rgb(path_or_rgb) if isinstance(path_or_rgb, (str, bytes)) or hasattr(path_or_rgb, "__fspath__") \
        else path_or_rgb
    return to_model_input(box_resize(rgb, size, size), lut)
