"""CLI inference: predict crop disease from a single image or a folder of
images, using a training checkpoint or an exported TorchScript/ONNX model.

Usage:
    python predict.py --model checkpoints/convnextv2_tiny_baseline/best.pt --image leaf.jpg
    python predict.py --model outputs/export/model.onnx --folder path/to/images/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging

from inference.predictor import DiseasePredictor
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crop disease from image(s).")
    parser.add_argument("--model", required=True, help="Path to a .pt checkpoint, .torchscript.pt, or .onnx model")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--folder", help="Path to a folder of images (non-recursive)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", help="Optional path to write results as JSON")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide either --image or --folder.")
    if args.image and args.folder:
        parser.error("Provide only one of --image or --folder, not both.")

    setup_logging("./logs", name="predict")
    predictor = DiseasePredictor(args.model)

    if args.image:
        result = predictor.predict(args.image, top_k=args.top_k)
        logger.info("Prediction: %s", json.dumps(result, indent=2))
    else:
        result = predictor.predict_folder(args.folder, top_k=args.top_k)
        logger.info("Predicted %d image(s).", len(result))
        for name, pred in result.items():
            logger.info("  %s -> %s (%.1f%%)", name, pred["disease"], pred["confidence"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
