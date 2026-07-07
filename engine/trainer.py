from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from configs.schema import Config
from engine.metrics import EpochMetrics, MetricTracker
from utils.checkpoint import save_checkpoint
from utils.device import get_amp_dtype, to_channels_last
from utils.logging_utils import EpochLogger
from utils.plots import plot_curve

logger = logging.getLogger(__name__)

class EarlyStopping:

    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0):
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.best: Optional[float] = None
        self.counter = 0

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        improved = self.is_improvement(value)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fn: nn.Module,
        device: torch.device,
        cfg: Config,
        class_to_idx: dict[str, int],
    ):
        if not hasattr(scheduler, "step_every"):
            raise AttributeError(
                "scheduler must have a `.step_every` attribute ('epoch' or 'batch') — "
                "build it via engine.scheduler.build_scheduler, not a raw torch scheduler."
            )
        valid_metric_names = set(EpochMetrics.__dataclass_fields__)
        if cfg.train.early_stopping_metric not in valid_metric_names:
            raise ValueError(
                f"train.early_stopping_metric={cfg.train.early_stopping_metric!r} is not a "
                f"field of EpochMetrics. Valid options: {sorted(valid_metric_names)}"
            )

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.cfg = cfg
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}

        self._amp_enabled = cfg.train.amp and device.type == "cuda"
        self._amp_dtype = get_amp_dtype(device) if self._amp_enabled else torch.float32
        self._use_grad_scaler = self._amp_enabled and self._amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(device.type, enabled=self._use_grad_scaler)
        if self._amp_enabled:
            logger.info("AMP enabled: dtype=%s, grad_scaler=%s", self._amp_dtype, self._use_grad_scaler)
        self.metric_tracker = MetricTracker(
            self.idx_to_class, min_eval_samples=cfg.train.early_stopping_min_eval_samples
        )
        self.early_stopping = EarlyStopping(
            patience=cfg.train.early_stopping_patience, mode=cfg.train.early_stopping_mode,
            min_delta=cfg.train.early_stopping_min_delta,
        )
        self.epoch_logger = EpochLogger(
            cfg.log.log_dir, cfg.log.experiment_name,
            use_tensorboard=cfg.log.tensorboard, use_csv=cfg.log.csv,
        )

        self.checkpoint_dir = Path(cfg.log.checkpoint_dir) / cfg.log.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_top1_accuracy": []}

    def _forward_batch(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        images = to_channels_last(images, enabled=self.cfg.train.channels_last)
        targets = targets.to(self.device, non_blocking=True)

        with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_enabled):
            logits = self.model(images)
            loss = self.loss_fn(logits, targets)
        return logits, loss

    def _backward_and_step(self, loss: torch.Tensor, accum_steps: int) -> None:
        scaled_loss = loss / accum_steps
        if self._use_grad_scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _optimizer_step(self) -> None:
        if self.cfg.train.grad_clip_norm > 0:
            if self._use_grad_scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)

        if self._use_grad_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler.step_every == "batch":
            self.scheduler.step()

    def _train_one_epoch(self) -> float:
        self.model.train()
        accum_steps = max(1, self.cfg.train.grad_accum_steps)
        running_loss, n_batches = 0.0, 0
        n_train_batches = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(self.train_loader):
            _, loss = self._forward_batch(images, targets)
            self._backward_and_step(loss, accum_steps)

            is_accum_boundary = (step + 1) % accum_steps == 0 or (step + 1) == n_train_batches
            if is_accum_boundary:
                self._optimizer_step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / max(1, n_batches)

    @torch.no_grad()
    def _validate_one_epoch(self) -> EpochMetrics:
        self.model.eval()
        self.metric_tracker.reset()

        for images, targets in self.val_loader:
            logits, loss = self._forward_batch(images, targets)
            self.metric_tracker.update(logits, targets.to(self.device), loss.item())

        return self.metric_tracker.compute()

    def fit(
        self,
        epochs: int,
        start_epoch: int = 0,
        best_metric: Optional[float] = None,
        early_stopping_counter: int = 0,
    ) -> dict[str, list[float]]:
        if best_metric is not None:
            self.early_stopping.best = best_metric
        self.early_stopping.counter = early_stopping_counter

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            train_loss = self._train_one_epoch()
            val_metrics = self._validate_one_epoch()

            if self.scheduler.step_every == "epoch":
                self.scheduler.step()

            elapsed = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            monitored_value = getattr(val_metrics, self.cfg.train.early_stopping_metric)

            self.epoch_logger.log_epoch(
                epoch,
                {
                    "train_loss": train_loss,
                    **{f"val_{k}": v for k, v in val_metrics.to_log_dict().items()},
                    "lr": current_lr,
                    "epoch_time_sec": elapsed,
                },
            )
            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_metrics.loss)
            self._history["val_top1_accuracy"].append(val_metrics.top1_accuracy)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_top1=%.4f | val_top3=%.4f | "
                "%s=%.4f | lr=%.2e | %.1fs",
                epoch + 1, epochs, train_loss, val_metrics.loss, val_metrics.top1_accuracy,
                val_metrics.top3_accuracy, self.cfg.train.early_stopping_metric, monitored_value,
                current_lr, elapsed,
            )

            is_best = self.early_stopping.step(monitored_value)
            checkpoint_kwargs = dict(
                model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler,
                epoch=epoch, best_metric=self.early_stopping.best, class_to_idx=self.class_to_idx,
                config=self.cfg, extra={"early_stopping_counter": self.early_stopping.counter},
            )

            if is_best:
                save_checkpoint(self.checkpoint_dir / "best.pt", **checkpoint_kwargs)

            if (epoch + 1) % max(1, self.cfg.train.checkpoint_interval) == 0:
                save_checkpoint(self.checkpoint_dir / "last.pt", **checkpoint_kwargs)

            if self.early_stopping.should_stop:
                logger.info(
                    "Early stopping at epoch %d: no improvement in '%s' for %d epochs.",
                    epoch + 1, self.cfg.train.early_stopping_metric, self.early_stopping.patience,
                )
                break

        self._save_curves()
        self.epoch_logger.close()
        return self._history

    def _save_curves(self) -> None:
        output_dir = Path(self.cfg.log.output_dir) / self.cfg.log.experiment_name / "plots"
        plot_curve(
            {"train": self._history["train_loss"], "val": self._history["val_loss"]},
            title="Loss", ylabel="Loss", path=output_dir / "loss_curve.png",
        )
        plot_curve(
            {"val_top1": self._history["val_top1_accuracy"]},
            title="Validation Top-1 Accuracy", ylabel="Accuracy", path=output_dir / "accuracy_curve.png",
        )
