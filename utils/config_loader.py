from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

from configs.schema import Config

logger = logging.getLogger(__name__)

T = TypeVar("T")

def _cast_scalar(raw: str, target_type: type) -> Any:
    if target_type is bool:
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    if target_type in (int, float, str):
        return target_type(raw)
    return yaml.safe_load(raw)

def _dataclass_from_dict(cls: Type[T], data: dict) -> T:
    if not dataclasses.is_dataclass(cls):
        return data  

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    unknown = set(data) - set(field_types)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = field_types[name]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)

def _apply_override(cfg: Config, dotted_key: str, raw_value: str) -> None:
    parts = dotted_key.split(".")
    obj = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise ValueError(f"Unknown config path in override '{dotted_key}' at '{part}'")
        obj = getattr(obj, part)

    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise ValueError(f"Unknown config path in override '{dotted_key}' at '{leaf}'")

    current = getattr(obj, leaf)
    target_type = type(current) if current is not None else str
    setattr(obj, leaf, _cast_scalar(raw_value, target_type))

def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = _dataclass_from_dict(Config, raw)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be in the form key.path=value")
        key, value = override.split("=", 1)
        _apply_override(cfg, key.strip(), value.strip())

    _validate(cfg)
    return cfg

def _warn_if_double_correcting(loss_name: str, use_sampler: bool, focal_gamma: float, context: str) -> None:
    is_reweighting_loss = loss_name == "weighted_ce" or (loss_name in {"focal", "focal_ls"} and focal_gamma > 0)
    if is_reweighting_loss and use_sampler:
        logger.warning(
            "[%s] loss.name=%r (with focal_gamma=%s) AND data.use_weighted_sampler=True are "
            "both enabled — both correct for the same class imbalance. Stacking them can "
            "overcorrect rare-class gradients into instability. Prefer one: the sampler "
            "(recommended default) OR a reweighting loss, not both — a head-only Stage 2 "
            "fine-tune (train.stage2) is the one case where combining them is low-risk, since "
            "a frozen backbone can't be destabilized by an aggressively rebalanced small head.",
            context, loss_name, focal_gamma,
        )

def _validate(cfg: Config) -> None:
    _warn_if_double_correcting(cfg.loss.name, cfg.data.use_weighted_sampler, cfg.loss.focal_gamma, "stage1")
    if cfg.train.stage2.enabled and not cfg.train.stage2.freeze_backbone:
        _warn_if_double_correcting(
            cfg.train.stage2.loss.name, cfg.data.use_weighted_sampler,
            cfg.train.stage2.loss.focal_gamma, "stage2 (backbone not frozen)",
        )

    total_holdout = cfg.data.val_split + cfg.data.test_split
    if not 0.0 < total_holdout < 1.0:
        raise ValueError(
            f"data.val_split + data.test_split must be in (0, 1), got {total_holdout}"
        )

    if cfg.data.min_samples_for_eval > cfg.data.min_samples_for_split:
        raise ValueError(
            "data.min_samples_for_eval must be <= data.min_samples_for_split"
        )

    if cfg.scheduler.name not in {"cosine", "cosine_warm_restarts", "onecycle"}:
        raise ValueError(f"Unknown scheduler.name: {cfg.scheduler.name}")

    valid_loss_names = {"ce", "weighted_ce", "focal", "label_smoothing", "focal_ls"}
    if cfg.loss.name not in valid_loss_names:
        raise ValueError(f"Unknown loss.name: {cfg.loss.name}")
    if cfg.train.stage2.loss.name not in valid_loss_names:
        raise ValueError(f"Unknown train.stage2.loss.name: {cfg.train.stage2.loss.name}")

def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)

def config_from_dict(data: dict) -> Config:
    cfg = _dataclass_from_dict(Config, data)
    _validate(cfg)
    return cfg

def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
