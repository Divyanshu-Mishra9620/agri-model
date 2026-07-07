from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from configs.schema import Config
from datasets.disease_dataset import CropDiseaseDataset
from datasets.transforms import build_eval_transforms, build_train_transforms
from utils.config_loader import load_config, save_config
from utils.device import get_device, prepare_model
from utils.seed import set_seed

logger = logging.getLogger(__name__)

def _load_split_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    splits_dir = Path(cfg.data.splits_dir)
    class_to_idx_path = splits_dir / "class_to_idx.json"
    train_manifest = splits_dir / "manifest_train.csv"
    val_manifest = splits_dir / "manifest_val.csv"

    for path in (class_to_idx_path, train_manifest, val_manifest):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `python scripts/prepare_splits.py` first "
                f"(after `python scripts/validate_dataset.py`)."
            )

    class_to_idx = json.loads(class_to_idx_path.read_text(encoding="utf-8"))
    return pd.read_csv(train_manifest), pd.read_csv(val_manifest), class_to_idx

def _build_loaders(
    cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame, class_to_idx: dict[str, int]
) -> tuple[DataLoader, DataLoader]:
    from datasets.sampler import build_weighted_sampler 

    train_dataset = CropDiseaseDataset(train_df, class_to_idx, transform=build_train_transforms(cfg.data, cfg.augmentation))
    val_dataset = CropDiseaseDataset(val_df, class_to_idx, transform=build_eval_transforms(cfg.data, cfg.augmentation))

    sampler = build_weighted_sampler(train_df, cfg.data.max_sample_weight_ratio) if cfg.data.use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=sampler is None,  
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        drop_last=True, 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    return train_loader, val_loader

def _run_stage1(cfg: Config, model, train_loader, val_loader, device, class_to_idx: dict[str, int]):
    from engine.losses import build_loss
    from engine.optimizer import build_optimizer
    from engine.scheduler import build_scheduler
    from engine.trainer import Trainer
    from utils.checkpoint import load_checkpoint, restore_training_state

    class_counts = train_loader.dataset.df["class_name"].value_counts().to_dict()
    loss_fn = build_loss(cfg.loss, class_counts=class_counts, class_to_idx=class_to_idx)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, epochs=cfg.train.epochs, steps_per_epoch=len(train_loader))

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, cfg, class_to_idx)

    start_epoch, best_metric, early_stopping_counter = 0, None, 0
    if cfg.train.resume_from:
        checkpoint = load_checkpoint(cfg.train.resume_from)
        if checkpoint["class_to_idx"] != class_to_idx:
            raise RuntimeError(
                f"Cannot resume from {cfg.train.resume_from}: its class_to_idx does not match "
                f"the current data.splits_dir/class_to_idx.json. If the dataset's class set "
                f"genuinely changed, this checkpoint can no longer be resumed — retrain from scratch."
            )
        start_epoch, best_metric = restore_training_state(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=trainer.scaler,
        )
        early_stopping_counter = checkpoint.get("early_stopping_counter", 0)
        logger.info("Resuming Stage 1 from %s at epoch %d.", cfg.train.resume_from, start_epoch)

    trainer.fit(
        epochs=cfg.train.epochs, start_epoch=start_epoch,
        best_metric=best_metric, early_stopping_counter=early_stopping_counter,
    )
    return trainer

def _run_stage2(cfg: Config, model, train_loader, val_loader, device, class_to_idx: dict[str, int]):
    from engine.losses import build_loss
    from engine.optimizer import build_optimizer
    from engine.scheduler import build_scheduler
    from engine.trainer import Trainer
    from models.convnext_v2 import freeze_backbone

    logger.info("Starting Stage 2: decoupled classifier re-training.")
    stage2_cfg = copy.deepcopy(cfg)
    stage2_cfg.loss = stage2_cfg.train.stage2.loss
    stage2_cfg.optimizer.lr = stage2_cfg.train.stage2.lr
    stage2_cfg.scheduler.warmup_epochs = stage2_cfg.train.stage2.scheduler_warmup_epochs
    stage2_cfg.scheduler.min_lr = stage2_cfg.train.stage2.scheduler_min_lr
    stage2_cfg.log.experiment_name = f"{cfg.log.experiment_name}_stage2"

    if cfg.train.stage2.freeze_backbone:
        freeze_backbone(model)

    stage2_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=cfg.data.batch_size,
        sampler=train_loader.sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        drop_last=True,
    )

    class_counts = train_loader.dataset.df["class_name"].value_counts().to_dict()
    loss_fn = build_loss(stage2_cfg.loss, class_counts=class_counts, class_to_idx=class_to_idx)
    optimizer = build_optimizer(model, stage2_cfg.optimizer, apply_head_lr_multiplier=False)
    scheduler = build_scheduler(
        optimizer, stage2_cfg.scheduler, epochs=cfg.train.stage2.epochs, steps_per_epoch=len(stage2_train_loader),
    )

    trainer = Trainer(model, stage2_train_loader, val_loader, optimizer, scheduler, loss_fn, device, stage2_cfg, class_to_idx)
    trainer.fit(epochs=cfg.train.stage2.epochs)
    logger.info("Stage 2 complete.")
    return trainer

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the crop disease classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[], help='e.g. --override train.epochs=30 optimizer.lr=1e-4')
    args = parser.parse_args()

    from models.convnext_v2 import build_model 
    from utils.logging_utils import setup_logging  
    from utils.system_info import log_system_info

    cfg = load_config(args.config, overrides=args.override)

    setup_logging(cfg.log.log_dir, name="train")
    log_system_info(Path(cfg.log.log_dir) / cfg.log.experiment_name)
    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    save_config(cfg, Path(cfg.log.output_dir) / cfg.log.experiment_name / "config_used.yaml")

    train_df, val_df, class_to_idx = _load_split_data(cfg)
    train_loader, val_loader = _build_loaders(cfg, train_df, val_df, class_to_idx)

    device = get_device()
    model = build_model(num_classes=len(class_to_idx), cfg=cfg.model)
    model = prepare_model(model, device, channels_last=cfg.train.channels_last, compile_model=cfg.train.torch_compile)

    _run_stage1(cfg, model, train_loader, val_loader, device, class_to_idx)

    if cfg.train.stage2.enabled:
        _run_stage2(cfg, model, train_loader, val_loader, device, class_to_idx)

if __name__ == "__main__":
    main()
