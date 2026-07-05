"""Export a trained checkpoint to TorchScript and ONNX, with a smoke test
confirming both exported artifacts produce (near-)identical output to the
original PyTorch model — so a broken export is caught here, not later
inside a serving process.

Run from the repo root: `python scripts/export_model.py --checkpoint checkpoints/<experiment>/best.pt`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from inference.predictor import load_model_from_checkpoint
from utils.config_loader import save_config
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def export_torchscript(model: torch.nn.Module, example_input: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, example_input)
    traced.save(str(output_path))
    logger.info("Saved TorchScript model: %s", output_path)


def export_onnx(model: torch.nn.Module, example_input: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Saved ONNX model: %s", output_path)


def smoke_test_torchscript(original: torch.nn.Module, path: Path, example_input: torch.Tensor) -> None:
    loaded = torch.jit.load(str(path))
    with torch.no_grad():
        original_out = original(example_input)
        loaded_out = loaded(example_input)
    if not torch.allclose(original_out, loaded_out, atol=1e-4):
        raise RuntimeError(f"TorchScript export mismatch at {path} — outputs differ beyond tolerance.")
    logger.info(
        "TorchScript export smoke test passed (max abs diff=%.2e).",
        (original_out - loaded_out).abs().max().item(),
    )


def smoke_test_onnx(original: torch.nn.Module, path: Path, example_input: torch.Tensor) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX smoke test (export was still saved).")
        return

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        original_out = original(example_input).numpy()
    onnx_out = session.run(None, {"image": example_input.numpy()})[0]

    if not np.allclose(original_out, onnx_out, atol=1e-3):
        raise RuntimeError(f"ONNX export mismatch at {path} — outputs differ beyond tolerance.")
    logger.info("ONNX export smoke test passed (max abs diff=%.2e).", np.abs(original_out - onnx_out).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to TorchScript and ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="./outputs/export")
    parser.add_argument("--formats", nargs="+", choices=["torchscript", "onnx"], default=["torchscript", "onnx"])
    args = parser.parse_args()

    setup_logging("./logs", name="export_model")

    model, cfg, class_to_idx = load_model_from_checkpoint(args.checkpoint)
    example_input = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "class_to_idx.json").open("w", encoding="utf-8") as fh:
        json.dump(class_to_idx, fh, indent=2, ensure_ascii=False)
    # Exported TorchScript/ONNX artifacts carry no config of their own —
    # inference/predictor.py::DiseasePredictor reads this back to reconstruct
    # the exact preprocessing (image size, normalization) the model was
    # trained with. Without it, serving would silently fall back to defaults,
    # which is wrong for any run that trained at a non-default image_size.
    save_config(cfg, output_dir / "config.yaml")

    if "torchscript" in args.formats:
        ts_path = output_dir / "model.torchscript.pt"
        export_torchscript(model, example_input, ts_path)
        smoke_test_torchscript(model, ts_path, example_input)

    if "onnx" in args.formats:
        onnx_path = output_dir / "model.onnx"
        export_onnx(model, example_input, onnx_path)
        smoke_test_onnx(model, onnx_path, example_input)

    logger.info("Export complete: %s", output_dir)


if __name__ == "__main__":
    main()
