from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:

    root_dir: str = "./data/raw"
    splits_dir: str = "./data/splits"
    clean_manifest: str = "./outputs/dataset_report/clean_manifest.csv"

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    val_split: float = 0.10
    test_split: float = 0.10
    min_samples_for_split: int = 10
    min_samples_for_eval: int = 3

    use_weighted_sampler: bool = True
    max_sample_weight_ratio: float = 20.0

@dataclass
class AugmentationConfig:

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

@dataclass
class ModelConfig:

    name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    drop_path_rate: float = 0.1
    num_classes: int = -1

@dataclass
class LossConfig:

    name: str = "label_smoothing"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    class_weight_power: float = 1.0

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    head_lr_multiplier: float = 5.0
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    no_weight_decay_on_norm_and_bias: bool = True

@dataclass
class SchedulerConfig:

    name: str = "cosine"
    warmup_epochs: int = 3
    min_lr: float = 1.0e-6
    warm_restarts_t0: int = 10
    warm_restarts_tmult: int = 2
    onecycle_pct_start: float = 0.1

@dataclass
class Stage2Config:

    enabled: bool = False
    epochs: int = 15
    freeze_backbone: bool = True
    loss: LossConfig = field(default_factory=lambda: LossConfig(name="label_smoothing", label_smoothing=0.1))
    lr: float = 1.0e-4
    scheduler_warmup_epochs: int = 1
    scheduler_min_lr: float = 1.0e-6

@dataclass
class TrainConfig:
    epochs: int = 60
    grad_accum_steps: int = 4
    grad_clip_norm: float = 1.0
    amp: bool = True
    channels_last: bool = True
    torch_compile: bool = False

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_metric: str = "macro_f1_eval_subset"
    early_stopping_mode: str = "max"
    early_stopping_min_eval_samples: int = 2

    checkpoint_interval: int = 1
    resume_from: Optional[str] = None

    seed: int = 42
    deterministic: bool = False

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
