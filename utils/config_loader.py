"""YAML <-> typed dataclass config loading with dotlist CLI overrides.

Deliberately hand-rolled instead of Hydra/OmegaConf: Hydra changes the
process's working directory per run and creates its own timestamped output
tree, which fights with the explicit checkpoints/ logs/ outputs/ layout this
project already uses. A ~100-line loader over `configs/schema.py` gives the
same "YAML + CLI overrides" ergonomics without that side effect.
"""

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
    """Cast a CLI-override string to the type of the field it's replacing."""
    if target_type is bool:
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    if target_type in (int, float, str):
        return target_type(raw)
    # list / Optional / anything else: let YAML's scalar parser decide
    # (handles "[0.9, 0.999]", "null", numbers, etc.)
    return yaml.safe_load(raw)


def _dataclass_from_dict(cls: Type[T], data: dict) -> T:
    """Recursively build a dataclass instance from a nested dict, validating
    that every key in `data` corresponds to a real field (fail fast on a
    typo'd YAML key instead of silently ignoring it)."""
    if not dataclasses.is_dataclass(cls):
        return data  # type: ignore[return-value]

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

    # Fields not present in `data` simply fall back to the dataclass default,
    # since we build via `cls(**kwargs)` rather than requiring every field.
    return cls(**kwargs)


def _apply_override(cfg: Config, dotted_key: str, raw_value: str) -> None:
    """Apply a single `section.field=value` CLI override in place."""
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
    """Load a YAML config into a validated `Config`, then apply CLI overrides.

    Args:
        path: Path to a YAML file (see configs/default.yaml for the full,
            commented reference).
        overrides: CLI-style `["train.lr=1e-4", "data.batch_size=16"]`
            dotlist overrides, applied after the YAML is loaded.
    """
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
    """`weighted_ce`, and `focal`/`focal_ls` with gamma>0, are themselves
    imbalance-correction mechanisms just like the sampler — reweighting by
    class frequency and by prediction confidence (which correlates strongly
    with class frequency once the sampler is active) respectively. Stacking
    either on top of the sampler double-corrects and can destabilize
    rare-class gradients, so this is a loud warning, not a silent default.
    """
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
    """Cross-field sanity checks that a per-field type doesn't catch."""
    _warn_if_double_correcting(cfg.loss.name, cfg.data.use_weighted_sampler, cfg.loss.focal_gamma, "stage1")
    if cfg.train.stage2.enabled and not cfg.train.stage2.freeze_backbone:
        # Stage 2 with an unfrozen backbone reintroduces exactly the
        # destabilization risk decoupled training is meant to avoid.
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
    """Serialize a Config back to a plain dict (for logging / checkpointing)."""
    return dataclasses.asdict(cfg)


def config_from_dict(data: dict) -> Config:
    """Rebuild a validated Config from a plain nested dict — the inverse of
    `config_to_dict`. Used to reconstruct the exact training configuration
    saved inside a checkpoint (evaluate.py, predict.py, scripts/export_model.py
    all need this — implemented once here rather than three times)."""
    cfg = _dataclass_from_dict(Config, data)
    _validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Write a Config to YAML, e.g. alongside a checkpoint for provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
