"""Typed configuration schema for the disease-classifier training pipeline.

Every tunable lives here as a dataclass field with an explicit default —
`configs/default.yaml` mirrors these defaults in a human-editable form.
Using plain dataclasses (rather than Hydra/OmegaConf, or Pydantic) keeps
config loading fully explicit and dependency-light: no implicit
working-directory changes, no hidden run directories, and IDEs can
type-check every `cfg.section.field` access. Cross-field validation lives
in `utils/config_loader.py::_validate` rather than on the dataclasses
themselves, which is the one place a Pydantic model would have been a
marginally better fit — noted here as a deliberate, considered trade-off,
not an oversight.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Deliberately NOT using `from __future__ import annotations` here: it would
# turn every field's type annotation into a plain string at runtime (PEP
# 563), which breaks utils/config_loader.py::_dataclass_from_dict — it
# introspects `dataclasses.fields(cls)[i].type` to decide whether a field is
# a nested dataclass to recurse into, and `dataclasses.is_dataclass("LossConfig")`
# (a string) is always False. Python 3.11 supports `X | None` natively, so
# there's no need for the future import to write modern-looking hints anyway.


@dataclass
class DataConfig:
    """Dataset location, splitting, and DataLoader performance settings."""

    root_dir: str = "./data/raw"
    """ImageFolder-style root: root_dir/<ClassName>/<image>.jpg"""
    splits_dir: str = "./data/splits"
    """Where manifest_{train,val,test}.csv and class_to_idx.json are written/read."""
    clean_manifest: str = "./outputs/dataset_report/clean_manifest.csv"
    """Output of scripts/validate_dataset.py; split.py reads this instead of
    re-scanning the raw folders, so corrupted images are skipped without
    ever being deleted from disk."""

    image_size: int = 224
    batch_size: int = 32
    """32 (x train.grad_accum_steps=4 -> effective 128) is a measured-safe
    default on a 6GB-class GPU (RTX 4050) under AMP + channels_last — see
    README "Hardware sizing" for the benchmarked memory table. 48 has
    headroom to try; 64 is a soft ceiling, not a confident default."""
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    val_split: float = 0.10
    test_split: float = 0.10
    min_samples_for_split: int = 10
    """Classes with >= this many images get a normal stratified 3-way split."""
    min_samples_for_eval: int = 3
    """Classes with fewer images than this are trained on but excluded from
    val/test entirely (too few samples to evaluate meaningfully)."""

    use_weighted_sampler: bool = True
    max_sample_weight_ratio: float = 20.0
    """Caps a class's WeightedRandomSampler weight at this multiple of the
    median class weight. Uncapped inverse-frequency weighting gives every
    class *exactly* equal total draw-mass regardless of size — on a
    ~270k-image epoch, a 5-image class would then be drawn ~1000+ times per
    epoch from only 5 unique images, which is memorization risk on exact
    pixels, not a normalization-statistics problem (ConvNeXt V2 has no
    BatchNorm to destabilize; it's LayerNorm/GRN throughout, which normalizes
    per-sample and carries no running batch statistics). The cap keeps rare
    classes strongly favored without letting any single one dominate."""


@dataclass
class AugmentationConfig:
    """Albumentations pipeline parameters (train-time only; val/test just
    resize + normalize). Probabilities are kept moderate on purpose: the
    disease signal itself is fine-grained leaf texture/color, so aggressive
    noise/dropout can destroy the very cues the model needs to learn."""

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    random_resized_crop_scale_min: float = 0.7
    hflip_p: float = 0.5
    rotate_limit: int = 20
    rotate_p: float = 0.5
    brightness_contrast_p: float = 0.3
    color_jitter_p: float = 0.2
    clahe_p: float = 0.15
    gauss_noise_p: float = 0.1
    coarse_dropout_p: float = 0.15
    coarse_dropout_max_holes: int = 4
    coarse_dropout_max_size_frac: float = 0.1
    """Max hole size as a fraction of image_size (keeps it scale-invariant)."""


@dataclass
class ModelConfig:
    """timm model selection. `num_classes=-1` is a sentinel meaning "auto-detect
    from the dataset manifest at runtime" — never set manually."""

    name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    drop_path_rate: float = 0.1
    num_classes: int = -1


@dataclass
class LossConfig:
    """`name` selects one of: ce | weighted_ce | focal | label_smoothing | focal_ls.

    Default `label_smoothing` (plain cross-entropy + label smoothing, no
    per-class reweighting in the loss) is deliberately the *simple* option:
    class imbalance is already being corrected once, by the
    WeightedRandomSampler (data.use_weighted_sampler). `weighted_ce` and
    `focal`/`focal_ls` are also imbalance-correction mechanisms — stacking
    either on top of the sampler double-corrects (see
    config_loader._validate, which warns loudly if you do this). Reserve
    `focal`/`weighted_ce` for a head-only Stage 2 fine-tune
    (train.stage2.enabled) instead, where a frozen backbone makes aggressive
    rebalancing far lower-risk. Label smoothing itself is independent of
    the imbalance question — it guards against label noise/visual ambiguity
    common in scraped leaf-disease photos and improves the calibration of
    the confidence score shown to farmers, so it stays on in every variant.
    """

    name: str = "label_smoothing"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    class_weight_power: float = 1.0
    """Exponent applied to inverse-frequency class weights (weighted_ce only).
    1.0 = full inverse frequency; <1.0 softens it."""


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    """Base LR, applied to the pretrained backbone."""
    head_lr_multiplier: float = 5.0
    """The classifier head is freshly initialized (random weights) while the
    backbone starts from strong ImageNet pretraining — training both at the
    same LR lets the random head's large early gradients backpropagate into
    and distort good pretrained features. The head gets `lr * head_lr_multiplier`
    instead; the backbone gets `lr`. This is standard discriminative-LR
    practice for transfer learning with a replaced head."""
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    no_weight_decay_on_norm_and_bias: bool = True
    """Excludes LayerNorm and bias params from weight decay — standard
    practice for ConvNeXt-family models."""


@dataclass
class SchedulerConfig:
    """`name` selects one of: cosine | cosine_warm_restarts | onecycle.

    Default `cosine` (linear warmup -> cosine decay) is the standard
    fine-tuning recipe for ConvNeXt-family backbones: warmup avoids early
    LR spikes destabilizing pretrained LayerNorm statistics, and a single
    smooth decay is more predictable than warm restarts (which is better
    suited to deliberately-cyclic from-scratch training) or OneCycle
    (tuned for aggressive from-scratch schedules, less standard for
    fine-tuning a converged pretrained backbone).
    """

    name: str = "cosine"
    warmup_epochs: int = 3
    min_lr: float = 1.0e-6
    warm_restarts_t0: int = 10
    warm_restarts_tmult: int = 2
    onecycle_pct_start: float = 0.1


@dataclass
class Stage2Config:
    """Optional Stage 2: decoupled classifier re-training (Kang et al.,
    2020) — off by default so the default pipeline matches a standard
    single-stage fine-tune. When enabled, Stage 1 trains the full network
    with the mild capped sampler + label smoothing (the safe defaults
    above); Stage 2 then freezes the backbone and retrains only the
    classifier head with more aggressive class-balancing. This is the
    single highest-leverage lever for this dataset's extreme imbalance: it
    separates *where* each correction acts, so an aggressively rebalanced
    loss/sampler combination can't destabilize the 28M-param backbone,
    because there's no backbone gradient left to destabilize — only a
    small linear head is being retrained.
    """

    enabled: bool = False
    epochs: int = 15
    freeze_backbone: bool = True
    loss: LossConfig = field(default_factory=lambda: LossConfig(name="weighted_ce", label_smoothing=0.1))
    lr: float = 1.0e-4
    scheduler_warmup_epochs: int = 1
    scheduler_min_lr: float = 1.0e-6


@dataclass
class TrainConfig:
    epochs: int = 60
    grad_accum_steps: int = 4
    """Physical batch (data.batch_size) x grad_accum_steps = effective batch.
    Default 32 x 4 = 128 effective."""
    grad_clip_norm: float = 1.0
    amp: bool = True
    """Mixed precision is auto-dtype: bf16 when the GPU supports it (no
    GradScaler needed — bf16 has fp32's exponent range, so it can't
    underflow the way fp16 can), falling back to fp16+GradScaler on older
    (pre-Ampere) GPUs. See utils/device.py::get_amp_dtype."""
    channels_last: bool = True
    torch_compile: bool = False
    """Off by default — not just "weak Windows support" but two concrete
    Windows-specific risks: (1) Windows Triton wheels come from a separate,
    narrower-coverage community build than upstream Linux and can silently
    mismatch your pinned torch version; (2) NVIDIA's WDDM driver model
    (used by GeForce laptop GPUs on Windows) enforces a ~2s hang-detection
    watchdog (TDR) that a long first-time Inductor compile can trip as a
    false-positive GPU reset. Enable only after an isolated smoke test."""

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    """A change smaller than this doesn't count as improvement — dampens
    single-validation-sample noise (see early_stopping_metric) from
    resetting the patience counter on a coin-flip."""
    early_stopping_metric: str = "macro_f1_eval_subset"
    """Must name a field on engine.metrics.EpochMetrics (selected via
    getattr, so the two can never silently drift apart). Defaults to
    macro-F1 restricted to classes with enough validation support — not
    plain accuracy (which, with 180 imbalanced classes, is dominated by
    majority classes and can look excellent while rare classes are never
    learned) and not unrestricted macro-F1 (every class excluded from val
    entirely, by min_samples_for_eval, would otherwise contribute a fixed
    zero forever, capping the achievable score regardless of model quality;
    and among classes with exactly 1 val sample, a single flip swings
    macro-F1 by 1/num_classes independent of real model quality)."""
    early_stopping_mode: str = "max"
    early_stopping_min_eval_samples: int = 2
    """Classes with fewer than this many val samples are excluded from
    macro_f1_eval_subset (but still reported in the full per-class
    breakdown) — see early_stopping_metric above."""

    checkpoint_interval: int = 1
    resume_from: Optional[str] = None

    seed: int = 42
    deterministic: bool = False
    """True enables torch.use_deterministic_algorithms + deterministic cuDNN
    (slower, fully reproducible). False (default) uses cuDNN benchmark mode
    for speed, seeded but not bit-exact reproducible."""

    stage2: Stage2Config = field(default_factory=Stage2Config)


@dataclass
class LogConfig:
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    experiment_name: str = "convnextv2_tiny_baseline"
    tensorboard: bool = True
    csv: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
