# KrishiNova Disease Classifier

A production-grade training pipeline for crop disease classification, built around **ConvNeXt V2 Tiny** (via `timm`) and designed for a real, extremely imbalanced dataset: ~180 classes, ~300,000 images, ranging from 7 images in the rarest class to 60,000+ in the largest.

This trains the model. It does not serve it — `predict.py` and the export scripts produce artifacts meant to be loaded by a separate serving layer (see [Integration with KrishiNova](#integration-with-krishinova) below).

```
Camera → Image → ConvNeXt V2 Tiny → Disease Prediction → Knowledge Base Lookup → Symptoms → Precautions → Mitigation → Prevention
                  └──────────── this repo ────────────┘   └────────────── a separate system ──────────────┘
```

## Contents

- [Installation](#installation)
- [Dataset preparation](#dataset-preparation)
- [Training](#training)
- [Resuming training](#resuming-training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Export (TorchScript / ONNX)](#export-torchscript--onnx)
- [Expected folder structure](#expected-folder-structure)
- [Design decisions](#design-decisions)
- [Hardware sizing](#hardware-sizing)
- [Troubleshooting](#troubleshooting)
- [Integration with KrishiNova](#integration-with-krishinova)

## Installation

Requires **Python 3.11** (matches the versions installed alongside this project; also works on 3.10+).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# Install PyTorch FIRST with the correct CUDA build — plain `pip install torch`
# from PyPI often resolves to a CPU-only wheel. Match the CUDA version to your
# driver (check with `nvidia-smi`); this repo was developed against CUDA 12.x
# wheels on a driver reporting CUDA 13.1 (newer drivers are backward-compatible
# with older CUDA-toolkit wheel builds):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

> **Watch for a silent torch upgrade.** `timm`/`albumentations` don't pin an exact torch version, but pip's resolver can still decide to bump torch to satisfy some transitive constraint when installing `requirements.txt` — silently swapping your CUDA build for a mismatched or CPU-only one. After the `pip install -r requirements.txt` step, re-run:
> ```bash
> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> ```
> If the version changed or `cuda.is_available()` flipped to `False`, re-run the exact `pip install torch torchvision --index-url ...` command above to pin it back — this is a known pip/PyTorch ecosystem rough edge, not specific to this repo, but worth checking every time you add a new dependency later too.

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Dataset preparation

Your raw dataset should be laid out as an `ImageFolder`:

```
data/raw/
  Apple___Black_rot/
    image1.jpg
    image2.jpg
  Apple___Apple_scab/
    ...
  Tomato___Late_blight/
    ...
```

Point `data.root_dir` in your config at this folder (default: `./data/raw`).

**1. Validate the dataset** — scans for corrupted/unreadable images and exact duplicates (via perceptual hashing), and reports the per-class distribution. Never deletes or modifies anything; writes a *clean manifest* that everything downstream reads instead of the raw folders.

```bash
python scripts/validate_dataset.py --config configs/default.yaml
```

Outputs land in `outputs/dataset_report/`: `clean_manifest.csv`, `corrupted_images.csv`, `duplicate_report.csv`, `class_distribution.csv` + `.png`, `dataset_report.json` / `.md`.

**2. Build train/val/test splits** — imbalance-aware: see [Design decisions](#design-decisions) for exactly how classes with 7 images are handled differently from classes with 60,000.

```bash
python scripts/prepare_splits.py --config configs/default.yaml
```

Writes `manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`, `class_to_idx.json`, and `split_report.json` to `data/splits/`.

> `class_to_idx.json` is the fixed source of truth every trained checkpoint's output indices are keyed against. Re-running this script after the dataset's class set has changed will refuse to overwrite an existing mapping unless you pass `--force` — doing so silently would invalidate every existing checkpoint's label mapping without any error at inference time.

**3. (Optional) Benchmark your DataLoader** — picks the right `data.num_workers` empirically instead of guessing, which matters more on Windows (spawn-based multiprocessing) than Linux:

```bash
python scripts/benchmark_dataloader.py --config configs/default.yaml
```

## Training

```bash
python train.py --config configs/default.yaml
```

Everything is configurable via YAML — see `configs/default.yaml` for the fully-commented reference, and `configs/schema.py` for the typed schema with rationale in each field's docstring. Override individual values without creating a new file:

```bash
python train.py --override train.epochs=30 optimizer.lr=1e-4 data.batch_size=16
```

This runs **Stage 1**: the full network, fine-tuned end-to-end with a class-balanced sampler and label-smoothed cross-entropy. If `train.stage2.enabled: true`, Stage 2 (decoupled classifier re-training) runs automatically afterward — see [Design decisions](#design-decisions).

Per-epoch metrics go to TensorBoard and a flat CSV under `logs/<experiment_name>/`:

```bash
tensorboard --logdir logs
```

Checkpoints (`best.pt`, `last.pt`) go to `checkpoints/<experiment_name>/`. Loss/accuracy curves go to `outputs/<experiment_name>/plots/`.

## Resuming training

```bash
python train.py --override train.resume_from=checkpoints/convnextv2_tiny_baseline/last.pt
```

Resuming restores model, optimizer, scheduler, AMP scaler state, RNG state (python/numpy/torch/CUDA), and the early-stopping patience counter — training continues exactly as if it had never stopped, not from a reset state.

**Resuming after an interruption (crash, reboot) with the same `train.epochs`** works exactly as expected — verified end-to-end: stop mid-run, resume, and it picks up at the correct epoch with the correct LR, best-metric, and patience count.

**Resuming to deliberately train *longer* than originally planned (raising `train.epochs`) needs a caveat.** The default `cosine` scheduler's warmup/decay shape is computed from `train.epochs` *at construction time*; resuming restores the scheduler's saved internal state on top of a freshly-built schedule shaped for the *new* `train.epochs` value, so the two can disagree. Concretely: `CosineAnnealingLR` is periodic and not clamped past its `T_max` — if you resume past where the original schedule had already reached its LR floor, further steps continue the cosine curve, which mathematically rises back up rather than staying flat at `min_lr` (confirmed by direct testing, not just theory). This is a general property of cosine-family schedulers under resume, not specific to this repo — the same surprise shows up in most training frameworks that resume scheduler state verbatim. If you want more total training, the simplest reliable approach is to set `train.epochs` to the larger number *from the start* and let early stopping decide when to actually stop, rather than resuming into an extended run after the fact.

## Evaluation

Full metric suite on the held-out test set — top-1/top-3 accuracy, precision/recall/F1 (macro & weighted), confusion matrix, per-class accuracy, and a many/medium/few-shot bucket breakdown (the standard long-tail-recognition reporting convention, so one macro number can't hide that rare classes are systematically worse served):

```bash
python evaluate.py --checkpoint checkpoints/convnextv2_tiny_baseline/best.pt
```

Writes `test_metrics.json`, `per_class_accuracy.csv`, `confusion_matrix.png`, and `worst_classes.png` to `outputs/<experiment_name>/evaluate/`.

## Inference

```bash
# Single image
python predict.py --model checkpoints/convnextv2_tiny_baseline/best.pt --image leaf.jpg

# A folder of images, written to a JSON file
python predict.py --model outputs/export/model.onnx --folder ./samples/ --output results.json
```

Output shape:

```json
{
  "disease": "Tomato___Late_blight",
  "confidence": 96.42,
  "top_k": [
    {"disease": "Tomato___Late_blight", "confidence": 96.42},
    {"disease": "Tomato___Early_blight", "confidence": 2.11},
    {"disease": "Potato___Late_blight", "confidence": 0.87}
  ],
  "inference_time_ms": 18.4
}
```

`inference/predictor.py::DiseasePredictor` is the reusable class behind this CLI — instantiate it once and call `.predict()` / `.predict_batch()` / `.predict_folder()` from any Python code (a FastAPI service, a notebook, a webcam loop) instead of shelling out to `predict.py` per request.

## Export (TorchScript / ONNX)

```bash
python scripts/export_model.py --checkpoint checkpoints/convnextv2_tiny_baseline/best.pt
```

Writes `model.torchscript.pt`, `model.onnx`, `class_to_idx.json`, and `config.yaml` to `outputs/export/` — the latter two are required sidecars `DiseasePredictor` reads to reconstruct exact class names and preprocessing when loading an exported artifact (not a raw checkpoint). Each export is smoke-tested against the original PyTorch model's output before being trusted (`torch.allclose` within tolerance) — a broken export fails here, not later inside a serving process.

## Expected folder structure

```
ml/
  configs/            # schema.py (typed dataclasses) + default.yaml (commented reference)
  data/
    raw/              # your ImageFolder-style dataset (not committed)
    splits/           # generated manifests + class_to_idx.json (not committed)
  datasets/           # Dataset class, transforms, imbalance-aware split + sampler
  models/             # ConvNeXt V2 Tiny builder + backbone-freeze/head-identification helpers
  engine/             # losses, optimizer, scheduler, metrics, Trainer
  utils/              # seed, device/AMP, checkpoint, logging, config loader, plots, system info
  scripts/            # validate_dataset, prepare_splits, benchmark_dataloader, export_model
  inference/          # DiseasePredictor (checkpoint / TorchScript / ONNX)
  tests/              # pytest unit tests for the highest-bug-risk logic
  checkpoints/        # best.pt / last.pt per experiment (not committed)
  outputs/            # plots, reports, exports (not committed)
  logs/               # TensorBoard + CSV logs (not committed)
  train.py
  evaluate.py
  predict.py
  requirements.txt
```

Run every script from this directory (`ml/`) — imports assume that as the working root.

## Design decisions

A few choices here are not the "obvious default," made deliberately for this exact dataset. Recorded here so the *why* survives longer than a commit message.

**Config system: dataclasses, not Hydra/OmegaConf/Pydantic.** Hydra's per-run working-directory changes and self-managed output tree would fight the explicit `checkpoints/`/`outputs/`/`logs/` layout this repo already uses. Pydantic (used elsewhere in this monorepo's RAG service) was a reasonable alternative for free validation, but for ~30 config keys a hand-rolled typed-dataclass loader with explicit cross-field validation (`utils/config_loader.py::_validate`) covers the same fail-fast guarantees without the extra dependency.

**Imbalance handling is layered, not single-mechanism.** With counts spanning 7 to 60,000 images (~8500:1), no single lever is enough:
- `datasets/split.py` splits *per class* (mathematically what stratification means) with three explicit size bands, so nothing crashes and nothing silently loses eval coverage. See its module docstring for the exact rules.
- `datasets/sampler.py`'s `WeightedRandomSampler` is capped (`data.max_sample_weight_ratio`, default 20x the median class weight) — uncapped inverse-frequency weighting gives every class *exactly* equal draw-mass regardless of size, which on a ~270k-image epoch would draw a 5-image class over 1,000 times per epoch from only 5 unique images: memorization risk, not a normalization-statistics problem (ConvNeXt V2 has no BatchNorm — it's LayerNorm/GRN throughout, which carries no running batch statistics to destabilize).
- The default loss (`label_smoothing`) deliberately does **not** also reweight by class — the sampler is already correcting for imbalance once. `weighted_ce` and `focal`/`focal_ls` are available but double-correct if combined with the sampler; `utils/config_loader.py` warns loudly if you do this.
- **Stage 2 (optional, `train.stage2.enabled`)** — decoupled classifier re-training (Kang et al., 2020): after Stage 1, freeze the backbone and retrain only the classifier head with a more aggressively rebalancing loss. This is the one case where combining an aggressive loss with rebalancing is low-risk, because there's no backbone gradient left for it to destabilize — only a small linear head. This is the single highest-leverage lever for this dataset's specific imbalance and is recommended once Stage 1 alone plateaus.

**Discriminative learning rates.** The classifier head is freshly initialized while the backbone starts from strong ImageNet pretraining; training both at the same LR lets the random head's large early gradients distort good pretrained features. The head trains at `optimizer.lr * optimizer.head_lr_multiplier` (default 5x); the backbone trains at `optimizer.lr`.

**Loss & scheduler defaults.** `label_smoothing` (not the more aggressive `focal_ls`) is the Stage 1 default for the double-correction reason above; label smoothing itself stays on regardless — it's independent of the imbalance question and guards against label noise in scraped leaf-disease photos while improving confidence calibration (this repo's output confidence is shown directly to farmers). The scheduler default is linear warmup into cosine decay — the standard fine-tuning recipe for ConvNeXt-family backbones; `cosine_warm_restarts` and `onecycle` are available but suited to different regimes (deliberately-cyclic and from-scratch training, respectively).

**Early stopping monitors `macro_f1_eval_subset`, not accuracy.** With 180 imbalanced classes, accuracy is dominated by majority classes and can look excellent while rare classes are never learned. Unrestricted macro-F1 has its own blind spot: classes excluded from validation entirely (`data.min_samples_for_eval`) would otherwise contribute a fixed near-zero score forever, and classes with exactly one validation sample can swing macro-F1 by `1/num_classes` on a single coin-flip prediction. `macro_f1_eval_subset` restricts the average to classes with `>= train.early_stopping_min_eval_samples` validation support; `train.early_stopping_min_delta` further dampens single-sample noise from resetting the patience counter. Every metric field is still logged in full for visibility — only the *stopping decision* uses the restricted, more honest signal.

**No multi-GPU (`nn.DataParallel`).** This repo's dev hardware has exactly one GPU, so a DataParallel path would ship completely untested — and PyTorch's own docs recommend `DistributedDataParallel` over it even when multiple GPUs *are* available (DataParallel is single-process and GIL-bound, strictly slower, with no offsetting benefit). See [Scaling to multiple GPUs](#scaling-to-multiple-gpus) for the real upgrade path, documented rather than implemented.

**`torch.compile` defaults off.** Beyond generically "weak Windows support": Windows Triton wheels come from a narrower-coverage community build that can silently mismatch your pinned torch version, and NVIDIA's WDDM driver model (used by GeForce laptop GPUs on Windows) enforces a hang-detection watchdog that a long first-time Inductor compile can trip as a false-positive GPU reset. Enable it only after an isolated smoke test.

## Hardware sizing

Benchmarked on an RTX 4050 Laptop GPU (6GB VRAM, Ada/compute 8.9) with AMP + `channels_last`, ConvNeXt-Tiny-scale model (~28M params):

| batch size | peak allocated | peak reserved | verdict |
|---|---|---|---|
| 32 | 2.41 GB | 2.67 GB | safe (default) |
| 48 | 3.44 GB | 3.67 GB | safe, more headroom to try |
| 64 | 4.40 GB | 4.75 GB | soft ceiling — thin margin once real augmentation overhead lands |
| 96 | 6.41 GB | 6.91 GB | exceeds this card's 6.14GB physical VRAM |
| 128 | 8.36 GB | 9.01 GB | unsafe |

Default is `data.batch_size=32` × `train.grad_accum_steps=4` (effective batch 128). On a 6GB-class card, don't extrapolate upward from generic "6-8GB card" advice — re-verify with your actual model/pipeline; this table is a starting point, not a guarantee.

**AMP dtype is automatic**, not configurable: bf16 when the GPU supports it (`torch.cuda.is_bf16_supported()` — true on Ampere/Ada and newer), falling back to fp16+GradScaler on older cards. bf16 has fp32's exponent range and can't underflow the way fp16 can, and needs no loss-scaling at all.

**Windows + WDDM note:** WDDM can silently spill an over-budget allocation into shared system memory instead of raising a clean OOM. In practice this shows up as a confusing 10-50x slowdown with no error, not a crash — "it didn't crash" is not proof a batch size is safe. Watch dedicated-vs-shared memory in `nvidia-smi` (or Task Manager's GPU tab) during your first real run at a new batch size, not just whether it completes.

### Scaling to multiple GPUs

Not implemented here (see [Design decisions](#design-decisions)). The real path is `DistributedDataParallel` via `torchrun`:

```bash
torchrun --nproc_per_node=<N> train.py --config configs/default.yaml
```

This would require: (1) process-group init (`torch.distributed.init_process_group`) and wrapping the model in `DDP` instead of the current single-process setup in `utils/device.py`; (2) a `DistributedSampler`-aware training loader — note `WeightedRandomSampler` does **not** compose with `DistributedSampler` out of the box, so a per-rank-sharded weighted sampler would need to be written; (3) rank-0-only checkpointing/logging to avoid every process writing the same files. On native Windows, NCCL has no support at all (Linux-only) — DDP there falls back to the slower, historically flakier `gloo` backend, so real multi-GPU training is a Linux/WSL target in practice.

## Troubleshooting

**`ModuleNotFoundError` for `configs`, `datasets`, etc.** — run scripts from the `ml/` directory. Root-level scripts (`train.py`, `evaluate.py`, `predict.py`) rely on Python's automatic insertion of the script's own directory onto `sys.path`; files under `scripts/` add an explicit `sys.path` shim for the same reason but still assume you're invoking `python scripts/foo.py` from `ml/`, not `foo.py` from inside `scripts/`.

**Windows: DataLoader workers crash or hang with `num_workers > 0`.** Windows uses spawn-based multiprocessing, which re-imports the entry-point module in each worker process. Every script that can end up with `num_workers > 0` already guards its executable code behind `if __name__ == "__main__":` — if you're importing these modules elsewhere with your own driver code, keep that guard.

**`torch.cuda.amp` deprecation warning.** This repo intentionally uses the long-stable `torch.cuda.amp.GradScaler` (still fully functional) rather than the newer unified `torch.amp` namespace, for compatibility certainty across PyTorch versions. Safe to ignore; harmless if your installed PyTorch prints a `FutureWarning` about it.

**`CoarseDropout` raises a `TypeError` about unexpected keyword arguments.** Your installed Albumentations predates the 1.4 API change (`num_holes_range`/`hole_height_range`/`hole_width_range` replaced the older `max_holes`/`min_holes`/... arguments this repo doesn't use). `pip install -U albumentations`.

**Training is much slower than expected with no error.** See the WDDM note in [Hardware sizing](#hardware-sizing) — check `nvidia-smi` for shared-vs-dedicated memory before assuming the batch size is fine just because nothing crashed.

**Class list changed and `prepare_splits.py` refuses to run.** This is intentional — see the note at the end of [Dataset preparation](#dataset-preparation). Use `--force` only when you mean to invalidate existing checkpoints and retrain from scratch.

## Integration with KrishiNova

This repo produces a model; it does not serve one. The wider KrishiNova app currently identifies crop diseases via third-party vision-LLM providers (Groq/Gemini/HuggingFace, in the Node backend's `disease-detection` module) — freeform text output, no fixed taxonomy. This pipeline's `class_to_idx.json` (the sorted class names discovered from your dataset's folder structure) becomes exactly that taxonomy going forward.

`predict.py` / `DiseasePredictor.predict()` output (`disease`, `confidence` as 0-100, `top_k`) is intentionally shaped to slot in as a drop-in replacement for the *identification* step in the app's existing flow — it does not and should not attempt to also produce symptoms/treatment/fertilizers/prevention text; that Knowledge Base Lookup step is a separate concern the app's existing RAG service (`Retrieval aug gen/rag/`) is already well-suited to handle, keyed off the predicted disease name. A future FastAPI service wrapping this pipeline would, at minimum:

1. Instantiate `DiseasePredictor` **once** at startup (not per-request — model load is the expensive part).
2. Accept an uploaded image, call `.predict()`.
3. Hand the predicted `disease` name to the existing knowledge-base/RAG lookup for symptoms/precautions/mitigation/prevention.
4. Return the combined result in whatever shape the Node backend's `Analysis` schema already expects (`detection.disease`, `confidence`, `recommendations.*`).

Nothing in this repo depends on that integration existing — it trains and evaluates completely standalone.
