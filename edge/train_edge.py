"""Train the gateway's on-device model by distilling the cloud ConvNeXt into a
small MobileNetV2.

    python -m edge.train_edge --teacher checkpoints/convnextv2_tiny_baseline_stage2/best.pt

Run it from the ml/ folder, inside the same virtualenv you trained the teacher
with. It reuses the teacher checkpoint's own config, so it reads the same
data/splits manifests, the same class_to_idx.json and the same augmentation.

How the distillation works, per batch:
  * the augmented image is produced at the teacher's size (224 px) exactly as
    in normal training;
  * the teacher sees it at 224 px, the student sees the same image shrunk to
    --img-size (160 px by default) - the resolution the gateway will run at;
  * the student is trained on a mix of the teacher's softened probabilities
    (what it learns most from) and the true label.

Checkpoints go to checkpoints/<name>/best.pt and last.pt; re-running with
--resume continues from last.pt (the run survives a laptop going to sleep).

No labelled dataset any more? Point it at any folders of leaf/crop photos:

    python -m edge.train_edge --images D:/leaf_photos D:/more_photos

The teacher labels every photo itself (see edge/unlabeled.py for what that
can and cannot do). 10% of the photos are held back, and "accuracy" becomes
agreement with the teacher on them.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.schema import Config
from datasets.disease_dataset import CropDiseaseDataset
from datasets.transforms import build_eval_transforms, build_train_transforms
from edge.student import DEFAULT_IMAGE_SIZE, DEFAULT_STUDENT, SUPPORTED_STUDENTS, build_student
from edge.unlabeled import HOLDOUT_LIST, TRAIN_LIST, UnlabeledImages, read_list, scan_images, split_holdout, write_list
from engine.metrics import EpochMetrics, MetricTracker
from utils.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from utils.config_loader import config_from_dict
from utils.device import get_amp_dtype, unwrap_model
from utils.seed import set_seed

logger = logging.getLogger("edge.train")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill the ConvNeXt teacher into the gateway's on-device model.")
    p.add_argument("--teacher", required=False, default="checkpoints/convnextv2_tiny_baseline_stage2/best.pt",
                   help="Teacher checkpoint (the cloud model).")
    p.add_argument("--images", nargs="+", default=None, metavar="DIR",
                   help="Train from folders of unlabelled leaf photos (searched recursively) instead of the "
                        "labelled dataset; the teacher labels them.")
    p.add_argument("--holdout", type=float, default=0.1,
                   help="With --images: fraction of photos held back to measure agreement with the teacher.")
    p.add_argument("--no-teacher", action="store_true",
                   help="Train the student on labels only (no distillation). Faster, usually less accurate.")
    p.add_argument("--student", default=DEFAULT_STUDENT, choices=SUPPORTED_STUDENTS)
    p.add_argument("--img-size", type=int, default=DEFAULT_IMAGE_SIZE,
                   help="Input size the gateway runs at (square). 160 is the default; 128 halves the compute.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.7, help="Weight of the distillation loss (0..1).")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=4, help="Early-stopping patience in epochs.")
    p.add_argument("--num-workers", type=int, default=None, help="Defaults to the teacher config's value.")
    p.add_argument("--name", default=None, help="Run name (default: edge_<student>_<img-size>).")
    p.add_argument("--resume", action="store_true", help="Continue from checkpoints/<name>/last.pt.")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Start the student from random weights instead of ImageNet (offline / tests).")
    p.add_argument("--limit-train", type=int, default=0, help="Use only N training images (smoke tests).")
    p.add_argument("--limit-val", type=int, default=0, help="Use only N validation images (smoke tests).")
    p.add_argument("--checkpoint-dir", default=None, help="Defaults to the teacher config's log.checkpoint_dir.")
    p.add_argument("--log-dir", default=None, help="Defaults to the teacher config's log.log_dir.")
    return p.parse_args(argv)


def _setup_logging(log_dir: Path, name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def load_teacher(path: str, device: torch.device):
    from models.convnext_v2 import build_model

    ckpt = load_checkpoint(path)
    cfg = config_from_dict(ckpt["config"])
    class_to_idx = ckpt["class_to_idx"]
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.pretrained = False  # the checkpoint has the weights; don't download ImageNet ones
    model = build_model(num_classes=len(class_to_idx), cfg=model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, class_to_idx


def load_splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    splits_dir = Path(cfg.data.splits_dir)
    paths = [splits_dir / "class_to_idx.json", splits_dir / "manifest_train.csv", splits_dir / "manifest_val.csv"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run this from the ml/ folder after scripts/prepare_splits.py.")
    class_to_idx = json.loads(paths[0].read_text(encoding="utf-8"))
    return pd.read_csv(paths[1]), pd.read_csv(paths[2]), class_to_idx


def kd_loss(student_logits, teacher_logits, targets, alpha: float, temperature: float, label_smoothing: float):
    """targets=None means unlabelled photos: the teacher's answer is all there is."""
    s = student_logits.float()
    zero = torch.zeros((), device=s.device)
    hard = F.cross_entropy(s, targets, label_smoothing=label_smoothing) if targets is not None else None
    if teacher_logits is None or alpha <= 0:
        if hard is None:
            raise ValueError("unlabelled photos need the teacher")
        return hard, hard.detach(), zero
    t = teacher_logits.float()
    T = temperature
    soft = F.kl_div(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1), reduction="batchmean") * (T * T)
    if hard is None:
        return soft, zero, soft.detach()
    return alpha * soft + (1.0 - alpha) * hard, hard.detach(), soft.detach()


def build_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_with_warmup(total_steps: int, warmup_steps: int, min_ratio: float = 0.01):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.no_grad()
def evaluate_agreement(student, teacher, loader, device, img_size: int, amp_dtype, amp_enabled) -> dict:
    """How often the student picks the teacher's answer (top-1), and how often
    the teacher's answer is among the student's top 3, on held-back photos."""
    student.eval()
    agree1 = agree3 = n = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        small = F.interpolate(images, size=(img_size, img_size), mode="bilinear", align_corners=False, antialias=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            t = teacher(images).float()
            s = student(small).float()
        t1 = t.argmax(1)
        agree1 += int((s.argmax(1) == t1).sum())
        agree3 += int((s.topk(min(3, s.shape[1]), dim=1).indices == t1[:, None]).any(1).sum())
        n += int(t1.numel())
    return {"agree_top1": agree1 / max(1, n), "agree_top3": agree3 / max(1, n), "n": n}


@torch.no_grad()
def evaluate(model, loader, device, idx_to_class, amp_dtype, amp_enabled, min_eval_samples: int) -> EpochMetrics:
    model.eval()
    tracker = MetricTracker(idx_to_class, min_eval_samples=min_eval_samples)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
        loss = F.cross_entropy(logits.float(), targets)
        tracker.update(logits, targets, loss.item())
    return tracker.compute()


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher, cfg, teacher_classes = (None, None, None)
    if args.images and args.no_teacher:
        raise SystemExit("--images needs the teacher: it is what labels the photos.")
    if args.no_teacher:
        # The teacher checkpoint still supplies the config and the class order.
        ckpt = load_checkpoint(args.teacher)
        cfg = config_from_dict(ckpt["config"])
        teacher_classes = ckpt["class_to_idx"]
        del ckpt
    else:
        teacher, cfg, teacher_classes = load_teacher(args.teacher, device)

    name = args.name or f"edge_{args.student}_{args.img_size}"
    log_dir = Path(args.log_dir or cfg.log.log_dir)
    _setup_logging(log_dir, name)
    set_seed(cfg.train.seed, deterministic=False)

    ckpt_dir = Path(args.checkpoint_dir or cfg.log.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    teacher_size = cfg.data.image_size
    train_data_cfg = copy.deepcopy(cfg.data)
    val_data_cfg = copy.deepcopy(cfg.data)
    val_data_cfg.image_size = args.img_size
    num_workers = cfg.data.num_workers if args.num_workers is None else args.num_workers
    unlabeled = bool(args.images)
    sampler = None

    if unlabeled:
        class_to_idx = teacher_classes
        if args.resume and (ckpt_dir / TRAIN_LIST).exists():
            train_paths, hold_paths = read_list(ckpt_dir / TRAIN_LIST), read_list(ckpt_dir / HOLDOUT_LIST)
        else:
            paths = scan_images(args.images)
            if len(paths) < 200:
                raise SystemExit(f"Only {len(paths)} photos found in {args.images}; use at least a few thousand.")
            train_paths, hold_paths = split_holdout(paths, args.holdout, seed=cfg.train.seed)
            write_list(ckpt_dir / TRAIN_LIST, train_paths)
            write_list(ckpt_dir / HOLDOUT_LIST, hold_paths)
        if args.limit_train:
            train_paths = train_paths[: args.limit_train]
        if args.limit_val:
            hold_paths = hold_paths[: args.limit_val]
        teacher_eval_cfg = copy.deepcopy(cfg.data)  # teacher-size eval transform; the student's copy is shrunk in the loop
        train_ds = UnlabeledImages(train_paths, transform=build_train_transforms(train_data_cfg, cfg.augmentation))
        val_ds = UnlabeledImages(hold_paths, transform=build_eval_transforms(teacher_eval_cfg, cfg.augmentation))
    else:
        train_df, val_df, class_to_idx = load_splits(cfg)
        if class_to_idx != teacher_classes:
            raise RuntimeError(
                "data/splits/class_to_idx.json does not match the teacher checkpoint's classes. "
                "The edge model must predict exactly the cloud model's labels."
            )
        if args.limit_train:
            train_df = train_df.sample(n=min(args.limit_train, len(train_df)), random_state=0).reset_index(drop=True)
        if args.limit_val:
            val_df = val_df.sample(n=min(args.limit_val, len(val_df)), random_state=0).reset_index(drop=True)
        missing = sum(1 for p in train_df["filepath"].head(50) if not Path(p).exists())
        if missing == min(50, len(train_df)):
            raise SystemExit(
                f"The training photos listed in {cfg.data.splits_dir}/manifest_train.csv are not on this "
                f"machine (e.g. {train_df['filepath'].iloc[0]}). Without them, train from any folder of "
                f"leaf photos instead: python -m edge.train_edge --images <folder>"
            )
        train_ds = CropDiseaseDataset(train_df, class_to_idx, transform=build_train_transforms(train_data_cfg, cfg.augmentation))
        val_ds = CropDiseaseDataset(val_df, class_to_idx, transform=build_eval_transforms(val_data_cfg, cfg.augmentation))
        if cfg.data.use_weighted_sampler:
            from datasets.sampler import build_weighted_sampler

            sampler = build_weighted_sampler(train_df, cfg.data.max_sample_weight_ratio)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    loader_kw = dict(num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers > 0:
        loader_kw.update(persistent_workers=cfg.data.persistent_workers, prefetch_factor=cfg.data.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    if len(train_loader) == 0:
        raise RuntimeError("Training set is smaller than one batch - lower --batch-size.")

    student = build_student(args.student, num_classes=len(class_to_idx), pretrained=not args.no_pretrained).to(device)
    if device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)

    amp_enabled = device.type == "cuda"
    amp_dtype = get_amp_dtype(device) if amp_enabled else torch.float32
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)

    optimizer = torch.optim.AdamW(build_param_groups(student, args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch))
    )

    # The checkpoint carries a Config so utils.checkpoint can save it; the edge
    # settings ride along in `extra` and are what export_edge.py reads back.
    run_cfg = copy.deepcopy(cfg)
    run_cfg.log.experiment_name = name
    edge_meta = {
        "mode": "images" if unlabeled else "labelled",
        "image_dirs": [str(d) for d in args.images] if unlabeled else None,
        "student": args.student,
        "img_size": args.img_size,
        "teacher": str(args.teacher),
        "teacher_img_size": teacher_size,
        "distilled": teacher is not None,
        "alpha": args.alpha,
        "temperature": args.temperature,
    }

    start_epoch, best_metric, bad_epochs = 0, None, 0
    if args.resume:
        last = ckpt_dir / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"--resume given but {last} does not exist.")
        ckpt = load_checkpoint(last)
        if ckpt.get("class_to_idx") != class_to_idx or ckpt.get("edge", {}).get("student") != args.student \
                or ckpt.get("edge", {}).get("img_size") != args.img_size:
            raise RuntimeError(f"{last} was trained with different classes / student / --img-size.")
        start_epoch, best_metric = restore_training_state(
            ckpt, model=student, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        bad_epochs = int(ckpt.get("early_stopping_counter", 0))
        if best_metric == float("-inf"):
            best_metric = None

    csv_path = log_dir / name / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Edge training (%s): student=%s img=%d teacher=%s (at %d px) epochs=%d batch=%d lr=%.1e alpha=%.2f T=%.1f "
        "train=%d val=%d device=%s",
        "unlabelled photos, teacher labels them" if unlabeled else "labelled dataset",
        args.student, args.img_size, "none" if teacher is None else args.teacher, teacher_size,
        args.epochs, args.batch_size, args.lr, 1.0 if unlabeled else args.alpha, args.temperature,
        len(train_ds), len(val_ds), device,
    )

    for epoch in range(start_epoch, args.epochs):
        student.train()
        t0 = time.time()
        sums = {"loss": 0.0, "hard": 0.0, "soft": 0.0}
        n = 0
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            small = F.interpolate(images, size=(args.img_size, args.img_size), mode="bilinear",
                                  align_corners=False, antialias=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                t_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(images)
                s_logits = student(small)
            loss, hard, soft = kd_loss(s_logits, t_logits, None if unlabeled else targets,
                                       args.alpha, args.temperature, args.label_smoothing)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                optimizer.step()
            scheduler.step()

            sums["loss"] += loss.item()
            sums["hard"] += hard.item()
            sums["soft"] += soft.item()
            n += 1
            if step % 200 == 0:
                logger.info("epoch %d step %d/%d loss=%.4f (hard %.4f, kd %.4f) lr=%.2e",
                            epoch + 1, step, steps_per_epoch, loss.item(), hard.item(), soft.item(),
                            optimizer.param_groups[0]["lr"])

        if unlabeled:
            val = evaluate_agreement(student, teacher, val_loader, device, args.img_size, amp_dtype, amp_enabled)
            monitored = val["agree_top1"]
            val_text = f"agrees with teacher top-1 {val['agree_top1']:.4f}, top-3 {val['agree_top3']:.4f} ({val['n']} held-back photos)"
            val_row = {"val_agree_top1": val["agree_top1"], "val_agree_top3": val["agree_top3"]}
        else:
            metrics = evaluate(student, val_loader, device, idx_to_class, amp_dtype, amp_enabled,
                               cfg.train.early_stopping_min_eval_samples)
            monitored = metrics.macro_f1_eval_subset
            val_text = (f"val_top1={metrics.top1_accuracy:.4f} val_top3={metrics.top3_accuracy:.4f} "
                        f"macro_f1={monitored:.4f}")
            val_row = {f"val_{k}": v for k, v in metrics.to_log_dict().items()}
        elapsed = time.time() - t0
        improved = best_metric is None or monitored > best_metric + 1e-4
        if improved:
            best_metric, bad_epochs = monitored, 0
        else:
            bad_epochs += 1

        logger.info(
            "Epoch %d/%d | train_loss=%.4f (hard %.4f, kd %.4f) | %s | %.0fs%s",
            epoch + 1, args.epochs, sums["loss"] / n, sums["hard"] / n, sums["soft"] / n,
            val_text, elapsed, "  * best" if improved else "",
        )
        row = {"epoch": epoch, "train_loss": sums["loss"] / n, **val_row,
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_sec": elapsed}
        new_file = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write(",".join(row.keys()) + "\n")
            fh.write(",".join(f"{v}" for v in row.values()) + "\n")

        kwargs = dict(model=unwrap_model(student), optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                      epoch=epoch, best_metric=best_metric, class_to_idx=class_to_idx, config=run_cfg,
                      extra={"early_stopping_counter": bad_epochs, "edge": edge_meta})
        if improved:
            save_checkpoint(ckpt_dir / "best.pt", **kwargs)
        save_checkpoint(ckpt_dir / "last.pt", **kwargs)

        if bad_epochs >= args.patience:
            logger.info("Early stopping: no improvement for %d epochs.", args.patience)
            break

    logger.info("Done. Best %s %.4f. Next: python -m edge.export_edge --checkpoint %s",
                "agreement with the teacher" if unlabeled else "val macro-F1",
                best_metric if best_metric is not None else float("nan"), ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
