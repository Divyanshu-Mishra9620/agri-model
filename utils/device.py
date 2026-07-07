from __future__ import annotations

import logging
import platform

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("CUDA available: using %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.warning("CUDA not available — training will run on CPU (very slow).")
    return torch.device("cpu")

def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)

def prepare_model(
    model: nn.Module,
    device: torch.device,
    *,
    channels_last: bool = True,
    compile_model: bool = False,
) -> nn.Module:
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
    if enabled and tensor.dim() == 4:
        return tensor.to(memory_format=torch.channels_last)
    return tensor
