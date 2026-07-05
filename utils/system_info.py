"""Reproducibility metadata: package versions, OS/GPU info, and (if available)
the current git commit hash — written once per run for provenance.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRACKED_MODULES = [
    "torch", "torchvision", "timm", "albumentations", "cv2", "numpy",
    "pandas", "sklearn", "PIL", "onnx", "onnxruntime",
]


def _module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def collect_system_info() -> dict[str, Any]:
    """Gather Python/OS/GPU/package-version info for the current process."""
    import torch  # local import so this module can be imported before torch elsewhere

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": gpus,
        "packages": {name: _module_version(name) for name in _TRACKED_MODULES},
    }


def log_system_info(log_dir: str | Path) -> dict[str, Any]:
    """Log system info and persist it as JSON under `log_dir/system_info.json`."""
    info = collect_system_info()
    logger.info("System info: %s", json.dumps(info, indent=2))

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "system_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    return info
