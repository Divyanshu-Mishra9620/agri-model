"""Device setup: CUDA/CPU selection, AMP dtype auto-detection, channels_last
memory format, and a guarded torch.compile wrapper that degrades gracefully.

Deliberately does NOT implement multi-GPU (nn.DataParallel): this repo's dev
hardware has exactly one GPU, so a DataParallel path would ship completely
untested, and PyTorch's own docs recommend DistributedDataParallel over it
even when it *is* exercised — DP is single-process and GIL-bound, strictly
slower, with no offsetting simplicity benefit worth shipping unexercised
code for. See README "Scaling to multiple GPUs" for the DDP+torchrun path
this repo documents instead of implementing.
"""

from __future__ import annotations

import logging
import platform

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Pick CUDA when available, otherwise fall back to CPU with a warning."""
    if torch.cuda.is_available():
        logger.info("CUDA available: using %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.warning("CUDA not available — training will run on CPU (very slow).")
    return torch.device("cpu")


def get_amp_dtype(device: torch.device) -> torch.dtype:
    """Prefer bf16 when the GPU supports it natively (Ampere/Ada and newer):
    bf16 has fp32's exponent range, so it can't underflow the way fp16 can
    and needs no loss-scaling GradScaler at all — one less moving part.
    Falls back to fp16 (which does need a GradScaler) on older GPUs, and is
    irrelevant on CPU (autocast is disabled there by the caller).
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped in torch.compile's OptimizedModule."""
    return getattr(model, "_orig_mod", model)


def prepare_model(
    model: nn.Module,
    device: torch.device,
    *,
    channels_last: bool = True,
    compile_model: bool = False,
) -> nn.Module:
    """Move `model` to `device` and apply the configured performance options.

    Each option is independently toggleable and safe to no-op when not
    applicable (e.g. channels_last is skipped on CPU).
    """
    model = model.to(device)

    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if compile_model:
        if platform.system() == "Windows":
            logger.warning(
                "train.torch_compile=True on Windows: the Triton/inductor "
                "backend has historically had weak Windows support, and "
                "WDDM's driver hang-watchdog (TDR) can mistake a long first "
                "compile for a hung GPU and force a reset. Attempting "
                "anyway; will fall back to eager mode on failure."
            )
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled.")
        except Exception:
            logger.exception("torch.compile failed — continuing in eager mode.")

    return model


def to_channels_last(tensor: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    """Convert an NCHW image batch to channels_last memory format when enabled."""
    if enabled and tensor.dim() == 4:
        return tensor.to(memory_format=torch.channels_last)
    return tensor
