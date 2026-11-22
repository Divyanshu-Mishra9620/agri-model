"""Turn the trained edge checkpoint into the gateway's int8 model.

    python -m edge.export_edge --checkpoint checkpoints/edge_mobilenetv2_050_160/best.pt

Steps:
  1. rebuild the student, fold BatchNorm, and check the folded graph against
     the PyTorch model;
  2. calibrate activation ranges on training photos run through the gateway's
     own preprocessing;
  3. quantize to int8 and measure on the held-out test set, all three ways:
       float model, normal eval resize        (what training reported)
       float model, gateway preprocessing     (isolates the resize change)
       int8 model,  gateway preprocessing     (what the gateway will score)
  4. write kn_edge_model.h / kn_edge_model.cpp into the gateway sketch folder
     (../../hardware/gateway_node by default) and a report into
     outputs/<run name>/.

The int8 numbers come from edge/int8.py's integer reference, which the
gateway reproduces bit for bit (it checks itself at boot with the self-test
vector written into kn_edge_model.cpp).

A model trained with `train_edge --images` (unlabelled photos) has no test
labels, so step 3 instead measures how often each of the three agrees with the
teacher on the photos that training held back.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import build_eval_transforms
from edge import int8
from edge.device_preprocess import box_resize, load_rgb, to_model_input
from edge.student import build_student
from edge.unlabeled import HOLDOUT_LIST, TRAIN_LIST, read_list
from engine.metrics import MetricTracker
from utils.checkpoint import load_checkpoint
from utils.config_loader import config_from_dict

logger = logging.getLogger("edge.export")


class DeviceInputDataset(Dataset):
    """Yields (gateway int8 input HWC, normal eval tensor CHW, label)."""

    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], size: int, lut: np.ndarray,
                 eval_transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.size = size
        self.lut = lut
        self.eval_transform = eval_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        try:
            rgb = load_rgb(row["filepath"])
        except FileNotFoundError:
            rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            label = -1
        else:
            label = self.class_to_idx[row["class_name"]]
        x_q = torch.from_numpy(to_model_input(box_resize(rgb, self.size, self.size), self.lut))
        x_std = self.eval_transform(image=rgb)["image"] if self.eval_transform is not None else torch.zeros(1)
        return x_q, x_std, label


class AgreementDataset(Dataset):
    """Unlabelled held-back photos: (gateway int8 input, student eval tensor, teacher eval tensor)."""

    def __init__(self, paths: list[str], size: int, lut: np.ndarray, student_tf, teacher_tf):
        self.paths, self.size, self.lut = paths, size, lut
        self.student_tf, self.teacher_tf = student_tf, teacher_tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            rgb = load_rgb(self.paths[i])
            ok = 1
        except FileNotFoundError:
            rgb, ok = np.zeros((self.size, self.size, 3), dtype=np.uint8), 0
        x_q = torch.from_numpy(to_model_input(box_resize(rgb, self.size, self.size), self.lut))
        return x_q, self.student_tf(image=rgb)["image"], self.teacher_tf(image=rgb)["image"], ok


@torch.no_grad()
def holdout_agreement(model, qm, meta, cfg, paths, lut, args, device) -> dict:
    from edge.train_edge import load_teacher

    teacher, teacher_cfg, _ = load_teacher(args.teacher or meta["teacher"], device)
    size = int(meta["img_size"])
    s_cfg = copy.deepcopy(cfg.data)
    s_cfg.image_size = size
    ds = AgreementDataset(paths, size, lut, build_eval_transforms(s_cfg, cfg.augmentation),
                          build_eval_transforms(teacher_cfg.data, teacher_cfg.augmentation))
    workers = cfg.data.num_workers if args.num_workers is None else args.num_workers
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)
    model = model.to(device)
    keys = ("float_eval_resize", "float_device_input", "int8_device_input")
    hits1 = dict.fromkeys(keys, 0)
    hits3 = dict.fromkeys(keys, 0)
    total = 0
    for x_q, x_s, x_t, ok in loader:
        keep = ok.bool()
        if not keep.any():
            continue
        x_q, x_s, x_t = x_q[keep].to(device), x_s[keep].to(device), x_t[keep].to(device)
        t1 = teacher(x_t).float().argmax(1)
        outs = {
            "float_eval_resize": model(x_s).float(),
            "float_device_input": model(int8.dequantize_input(qm, x_q)).float(),
            "int8_device_input": int8.logits_from_acc(qm, int8.int_forward(qm, x_q)).float(),
        }
        for k, lg in outs.items():
            hits1[k] += int((lg.argmax(1) == t1).sum())
            hits3[k] += int((lg.topk(min(3, lg.shape[1]), 1).indices == t1[:, None]).any(1).sum())
        total += int(t1.numel())
    result = {k: {"agree_top1": round(hits1[k] / max(1, total), 6), "agree_top3": round(hits3[k] / max(1, total), 6)}
              for k in keys}
    result["num_holdout_images"] = total
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize the edge model and emit C for the gateway.")
    p.add_argument("--checkpoint", required=True, help="checkpoints/<edge run>/best.pt from edge.train_edge")
    p.add_argument("--gateway-dir", default="../../hardware/gateway_node",
                   help="Gateway sketch folder to write kn_edge_model.* into (skipped if it does not exist).")
    p.add_argument("--out", default=None, help="Report folder (default outputs/<run name>).")
    p.add_argument("--calib-images", type=int, default=512)
    p.add_argument("--calib-method", choices=("minmax", "mean_minmax"), default="minmax")
    p.add_argument("--test-limit", type=int, default=0, help="Evaluate on N test images only (0 = all).")
    p.add_argument("--skip-eval", action="store_true", help="Do not measure test accuracy (much faster).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--teacher", default=None,
                   help="Cloud model checkpoint for the agreement check of an --images run "
                        "(default: the path recorded at training time).")
    return p.parse_args(argv)


def load_student(path: str):
    ckpt = load_checkpoint(path)
    meta = ckpt.get("edge")
    if not meta:
        raise RuntimeError(f"{path} is not an edge checkpoint (no 'edge' section) - train it with edge.train_edge")
    cfg = config_from_dict(ckpt["config"])
    class_to_idx = ckpt["class_to_idx"]
    model = build_student(meta["student"], num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), cfg, class_to_idx, meta, ckpt


def calibration_batches(df: pd.DataFrame, size: int, lut: np.ndarray, qm_in: int8.QParams, n: int,
                        batch: int, device: torch.device):
    sample = df.sample(n=min(n, len(df)), random_state=0)
    xs = []
    for path in sample["filepath"]:
        try:
            rgb = load_rgb(path)
        except FileNotFoundError:
            continue
        xs.append(to_model_input(box_resize(rgb, size, size), lut))
        if len(xs) == batch:
            yield _deq(np.stack(xs), qm_in, device)
            xs = []
    if xs:
        yield _deq(np.stack(xs), qm_in, device)


def _deq(x_q: np.ndarray, qp: int8.QParams, device: torch.device) -> torch.Tensor:
    x = (torch.from_numpy(x_q).to(torch.float64) - qp.zp) * qp.scale
    return x.permute(0, 3, 1, 2).contiguous().to(device)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, class_to_idx, meta, ckpt = load_student(args.checkpoint)
    size = int(meta["img_size"])
    labels = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    run_name = Path(args.checkpoint).parent.name
    out_dir = Path(args.out or Path(cfg.log.output_dir) / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = tuple(cfg.augmentation.mean), tuple(cfg.augmentation.std)

    logger.info("Student %s at %d px, %d classes, trained %s epochs (best val macro-F1 %.4f)",
                meta["student"], size, len(labels), ckpt.get("epoch", -1) + 1, ckpt.get("best_metric", float("nan")))

    graph = int8.extract_graph(model, size)
    rel = int8.verify_graph(graph, model, size)
    logger.info("Folded graph: %d ops, matches the PyTorch model to %.2e (relative)", len(graph.ops), rel)
    if rel > 1e-6:
        raise RuntimeError(f"folded graph disagrees with the model ({rel:.2e}) - refusing to export")

    in_qp = int8.input_qparams(mean, std)
    lut = int8.make_input_lut(mean, std, in_qp)
    splits_dir = Path(cfg.data.splits_dir)
    unlabeled = meta.get("mode") == "images"
    ckpt_dir = Path(args.checkpoint).parent
    if unlabeled:
        train_df = pd.DataFrame({"filepath": read_list(ckpt_dir / TRAIN_LIST)})
    else:
        train_df = pd.read_csv(splits_dir / "manifest_train.csv")
    t0 = time.time()
    obs = int8.calibrate(
        graph,
        calibration_batches(train_df, size, lut, in_qp, args.calib_images, 32, device),
        method=args.calib_method,
    )
    qm = int8.quantize(graph, obs, mean, std)
    logger.info("Calibrated on %d training photos (%s) in %.0fs", min(args.calib_images, len(train_df)),
                args.calib_method, time.time() - t0)

    report = {
        "checkpoint": str(args.checkpoint),
        "student": meta["student"],
        "img_size": size,
        "num_classes": len(labels),
        "calib_images": min(args.calib_images, len(train_df)),
        "calib_method": args.calib_method,
        "graph_check_rel_err": rel,
    }

    top1 = top3 = 0.0
    if unlabeled and not args.skip_eval:
        paths = read_list(ckpt_dir / HOLDOUT_LIST)
        if args.test_limit:
            paths = paths[: args.test_limit]
        t0 = time.time()
        agreement = holdout_agreement(model, qm, meta, cfg, paths, lut, args, device)
        report["holdout_agreement_with_teacher"] = agreement
        logger.info("Agreement with the cloud model on %d held-back photos (%.0fs) - no labels, so no accuracy:",
                    agreement["num_holdout_images"], time.time() - t0)
        for key in ("float_eval_resize", "float_device_input", "int8_device_input"):
            r = agreement[key]
            logger.info("  %-20s same top-1 %.2f%%  cloud's answer in top-3 %.2f%%", key,
                        r["agree_top1"] * 100, r["agree_top3"] * 100)
    elif not args.skip_eval:
        test_df = pd.read_csv(splits_dir / "manifest_test.csv")
        if args.test_limit:
            test_df = test_df.sample(n=min(args.test_limit, len(test_df)), random_state=0)
        eval_cfg = copy.deepcopy(cfg.data)
        eval_cfg.image_size = size
        ds = DeviceInputDataset(test_df, class_to_idx, size, lut, build_eval_transforms(eval_cfg, cfg.augmentation))
        workers = cfg.data.num_workers if args.num_workers is None else args.num_workers
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        trackers = {k: MetricTracker(idx_to_class, cfg.train.early_stopping_min_eval_samples)
                    for k in ("float_eval_resize", "float_device_input", "int8_device_input")}
        agree = total = 0
        model_dev = model.to(device)
        t0 = time.time()
        with torch.no_grad():
            for x_q, x_std, y in loader:
                keep = y >= 0
                if not keep.any():
                    continue
                x_q, x_std, y = x_q[keep].to(device), x_std[keep].to(device), y[keep].to(device)
                logit_std = model_dev(x_std).float()
                logit_fdev = model_dev(int8.dequantize_input(qm, x_q)).float()
                logit_q = int8.logits_from_acc(qm, int8.int_forward(qm, x_q)).float()
                for key, lg in (("float_eval_resize", logit_std), ("float_device_input", logit_fdev),
                                ("int8_device_input", logit_q)):
                    trackers[key].update(lg, y, torch.nn.functional.cross_entropy(lg, y).item())
                agree += int((logit_q.argmax(1) == logit_fdev.argmax(1)).sum())
                total += int(y.numel())
        results = {}
        for key, tr in trackers.items():
            m = tr.compute()
            results[key] = {k: round(float(v), 6) for k, v in m.to_log_dict().items()}
        results["int8_vs_float_top1_agreement"] = round(agree / max(1, total), 6)
        results["num_test_images"] = total
        report["test"] = results
        top1 = results["int8_device_input"]["top1_accuracy"]
        top3 = results["int8_device_input"]["top3_accuracy"]
        logger.info("Test set (%d images, %.0fs):", total, time.time() - t0)
        for key in ("float_eval_resize", "float_device_input", "int8_device_input"):
            r = results[key]
            logger.info("  %-20s top1 %.2f%%  top3 %.2f%%  macro-F1 %.2f%%", key,
                        r["top1_accuracy"] * 100, r["top3_accuracy"] * 100, r["f1_macro"] * 100)
        logger.info("  int8 agrees with float on %.2f%% of photos", results["int8_vs_float_top1_agreement"] * 100)

    targets = [out_dir / "c"]
    gw = Path(args.gateway_dir)
    if gw.is_dir():
        targets.append(gw)
    else:
        logger.warning("Gateway folder %s not found - C files only written to %s", gw, out_dir / "c")
    summary = None
    for target in targets:
        summary = int8.emit_c(qm, labels, target, student=meta["student"], test_top1=top1, test_top3=top3)
        logger.info("Wrote %s and %s", target / "kn_edge_model.h", target / "kn_edge_model.cpp")
    report.update(summary or {})
    report["written_to"] = [str(t) for t in targets]
    torch.save({"qm": qm, "labels": labels, "model_id": report["model_id"]}, out_dir / "edge_int8.pt")

    (out_dir / "edge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Model %s: %d layers, %.2f MMAC per photo, %.0f KB weights, %.0f KB activation arena",
        report["model_id"], report["layers"], report["macs"] / 1e6, report["weight_bytes"] / 1024,
        report["arena_bytes"] / 1024,
    )
    logger.info("Report: %s", out_dir / "edge_report.json")


if __name__ == "__main__":
    main()
