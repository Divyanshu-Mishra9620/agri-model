"""Run the exported int8 model on photos exactly the way the gateway does.

    python -m edge.predict_edge photo.jpg [photo2.jpg ...] --model outputs/edge_mobilenetv2_050_160/edge_int8.pt

Useful to check a capture: download the photo from the dashboard, run this,
and the top-3 should match the "on-device" result the gateway reported for it
(to within the tiny differences between OpenCV's JPEG decoder and the
gateway's own).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from edge import int8
from edge.device_preprocess import box_resize, load_rgb, to_model_input


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("images", nargs="+")
    p.add_argument("--model", required=True, help="edge_int8.pt written by edge.export_edge")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args(argv)

    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    qm: int8.QModel = bundle["qm"]
    labels = bundle["labels"]
    size = qm.img_size
    print(f"model {bundle['model_id']} ({size}x{size}, {len(labels)} classes)")
    for path in args.images:
        x = to_model_input(box_resize(load_rgb(path), size, size), qm.input_lut)
        acc = int8.int_forward(qm, torch.from_numpy(x[None]))
        logits = int8.logits_from_acc(qm, acc)[0].numpy()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = np.argsort(-probs)[: args.top_k]
        print(path)
        for i in top:
            print(f"  {probs[i] * 100:6.2f}%  {labels[i]}")


if __name__ == "__main__":
    main()
