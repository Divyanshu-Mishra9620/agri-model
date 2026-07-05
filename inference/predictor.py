"""Inference: load a trained model (checkpoint or exported TorchScript/ONNX
artifact) and run predictions on single images, a folder, or a batch.

This is the one place "load a trained model" and "preprocess + predict" are
implemented — scripts/export_model.py, evaluate.py, and predict.py all
reuse `load_model_from_checkpoint` / `DiseasePredictor` from here rather
than each re-implementing their own copy.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from configs.schema import Config
from datasets.transforms import build_eval_transforms
from models.convnext_v2 import build_model
from utils.checkpoint import load_checkpoint
from utils.config_loader import config_from_dict, load_config
from utils.device import get_device

logger = logging.getLogger(__name__)

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model_from_checkpoint(checkpoint_path: str | Path) -> tuple[torch.nn.Module, Config, dict[str, int]]:
    """Rebuild an eval-mode model exactly as it was trained, from a
    checkpoint saved by utils.checkpoint.save_checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = config_from_dict(checkpoint["config"])
    class_to_idx = checkpoint["class_to_idx"]

    model = build_model(num_classes=len(class_to_idx), cfg=cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, class_to_idx


class DiseasePredictor:
    """Loads a trained model once and serves predictions.

    Accepts either a raw `.pt` training checkpoint or an exported
    `.torchscript.pt` / `.onnx` artifact — the right backend is chosen from
    the file extension. Preprocessing always goes through
    `datasets.transforms.build_eval_transforms`, the exact same function
    used for validation during training, so serving can never silently
    drift from what the model was evaluated against.
    """

    def __init__(self, model_path: str | Path, device: torch.device | None = None):
        model_path = Path(model_path)
        self.device = device or get_device()
        self.backend, self.model, cfg, self.class_to_idx = self._load(model_path)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.transform = build_eval_transforms(cfg.data, cfg.augmentation)
        logger.info(
            "DiseasePredictor ready: backend=%s, %d classes, device=%s",
            self.backend, len(self.class_to_idx), self.device,
        )

    def _load(self, model_path: Path):
        suffix = "".join(model_path.suffixes)  # captures ".torchscript.pt" as one unit, not just ".pt"

        if suffix.endswith(".onnx"):
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(model_path), providers=providers)
            class_to_idx = self._load_sibling_class_to_idx(model_path)
            cfg = self._load_sibling_config(model_path)
            return "onnx", session, cfg, class_to_idx

        if suffix.endswith(".torchscript.pt"):
            model = torch.jit.load(str(model_path), map_location=self.device)
            model.eval()
            class_to_idx = self._load_sibling_class_to_idx(model_path)
            cfg = self._load_sibling_config(model_path)
            return "torchscript", model, cfg, class_to_idx

        model, cfg, class_to_idx = load_model_from_checkpoint(model_path)
        model = model.to(self.device).eval()
        return "checkpoint", model, cfg, class_to_idx

    @staticmethod
    def _load_sibling_class_to_idx(model_path: Path) -> dict[str, int]:
        import json

        sidecar = model_path.parent / "class_to_idx.json"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"Expected {sidecar} alongside the exported model (written by "
                f"scripts/export_model.py) — it's needed to map predictions back to class names."
            )
        return json.loads(sidecar.read_text(encoding="utf-8"))

    @staticmethod
    def _load_sibling_config(model_path: Path) -> Config:
        # Exported artifacts carry no config of their own — scripts/export_model.py
        # writes this sidecar precisely so preprocessing (image size,
        # normalization) can be reconstructed exactly, instead of silently
        # falling back to defaults that would be wrong for any model trained
        # at a non-default image_size (a train/serve skew bug, not a detail).
        sidecar = model_path.parent / "config.yaml"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"Expected {sidecar} alongside the exported model (written by "
                f"scripts/export_model.py) — it's needed to reconstruct the exact "
                f"preprocessing the model was trained with."
            )
        return load_config(sidecar)

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        image_path = str(image)
        array = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if array is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        if self.backend == "onnx":
            logits = self.model.run(None, {"image": batch.numpy()})[0]
            return logits
        with torch.no_grad():
            batch = batch.to(self.device)
            logits = self.model(batch)
            return logits.cpu().numpy()

    def predict(self, image: Union[str, Path, np.ndarray], top_k: int = 3) -> dict:
        """Predict on a single image. Returns disease name, confidence
        (0-100, matching this monorepo's existing Analysis schema
        convention), top-k predictions, and inference latency in ms.
        """
        start = time.perf_counter()

        array = self._load_image(image)
        tensor = self.transform(image=array)["image"].unsqueeze(0)
        logits = self._forward(tensor)

        probs = _softmax(logits[0])
        top_indices = np.argsort(probs)[::-1][:top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "disease": self.idx_to_class[int(top_indices[0])],
            "confidence": round(float(probs[top_indices[0]]) * 100, 2),
            "top_k": [
                {"disease": self.idx_to_class[int(idx)], "confidence": round(float(probs[idx]) * 100, 2)}
                for idx in top_indices
            ],
            "inference_time_ms": round(elapsed_ms, 2),
        }

    def predict_batch(self, images: list[Union[str, Path, np.ndarray]], top_k: int = 3) -> list[dict]:
        """Predict on a list of images in one forward pass (more efficient
        than calling `predict` in a loop for more than a handful of images)."""
        start = time.perf_counter()

        arrays = [self._load_image(image) for image in images]
        tensors = torch.stack([self.transform(image=arr)["image"] for arr in arrays])
        logits = self._forward(tensors)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_image_ms = elapsed_ms / max(1, len(images))

        results = []
        for row in logits:
            probs = _softmax(row)
            top_indices = np.argsort(probs)[::-1][:top_k]
            results.append(
                {
                    "disease": self.idx_to_class[int(top_indices[0])],
                    "confidence": round(float(probs[top_indices[0]]) * 100, 2),
                    "top_k": [
                        {"disease": self.idx_to_class[int(idx)], "confidence": round(float(probs[idx]) * 100, 2)}
                        for idx in top_indices
                    ],
                    "inference_time_ms": round(per_image_ms, 2),
                }
            )
        return results

    def predict_folder(self, folder: str | Path, top_k: int = 3) -> dict[str, dict]:
        """Predict on every image in a folder (non-recursive). Returns a
        dict keyed by filename."""
        folder = Path(folder)
        image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in _VALID_EXTENSIONS)
        if not image_paths:
            raise ValueError(f"No images found in {folder}")

        predictions = self.predict_batch(image_paths, top_k=top_k)
        return {path.name: pred for path, pred in zip(image_paths, predictions)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()
