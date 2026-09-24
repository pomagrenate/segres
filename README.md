# SOAR: Resolution-Preserving Scientific Segmentation Framework

A dedicated, high-resolution scientific semantic segmentation framework engineered specifically for native full-resolution imagery (2048x2048+) with extreme class imbalance and ultra-thin, curvilinear topological structures (e.g., solar filaments, retinal vessels, material cracks).

## Architectural Principles

- **Strict Native Resolution ($2048 \times 2048$)**: Enforces physical `batch_size = 1` during training, validation, and inference to prevent downsampling loss and eliminate background gradient swamping.
- **Virtual Batch Scaling**: Employs `--accumulate-grad-batches K` to simulate larger effective batch sizes without peak VRAM penalties.
- **Normalization Stability**: Built exclusively with dynamic `GroupNorm` layers, avoiding the variance drift and DDP desynchronization common to `BatchNorm2d` under single-sample batches.
- **Curvilinear Topology Optimization**: Composite objective combining continuous Signed Distance Field (SDF) boundary loss, warm-started topological `CLDiceLoss`, and numerically stable `DiceBCELoss`.
- **Decoupled Declarative Design**: Pure PyTorch modular design parameterized via YAML configurations across scales (`n`, `s`, `m`, `l`, `x`).

## Project Structure

```
OPT-HQ-Net/
├── configs/                     # Centralized declarative configurations
│   ├── default.yaml             # Master default training configuration
│   └── models/                  # Resolution-aware model specifications
│       ├── soar_nano1.yaml      # SOAR1-Nano (~0.94M)
│       ├── soar_small1.yaml     # SOAR1-Small (~2.89M)
│       ├── soar_medium1.yaml    # SOAR1-Medium (~6.80M, Default)
│       ├── soar_large1.yaml     # SOAR1-Large (~13.47M)
│       └── soar_xlarge1.yaml    # SOAR1-XLarge (~22.64M)
│
├── soar/                        # Core library namespace (installable PEP-517 package)
│   ├── __init__.py              # Public library exports
│   ├── cli.py                   # Unified CLI implementation
│   ├── data/                    # Scientific dataset, augmentations & preprocessing
│   │   ├── dataset.py           # Zero-copy streaming dataset loader
│   │   ├── augment.py           # Geometry-preserving data augmentations
│   │   ├── preprocess.py        # Scientific image preprocessing (CLAHE, percentile)
│   │   └── config.py            # Preprocessing configuration dataclasses
│   ├── engine/                  # Runtime execution engines
│   │   ├── trainer.py           # DDP trainer with virtual gradient accumulation
│   │   ├── validator.py         # O(1) streaming metric accumulator
│   │   └── predictor.py         # High-resolution streaming inference engine
│   ├── losses/                  # Numerically stable FP16/FP32 losses
│   │   ├── base.py              # Loss base class interface
│   │   ├── region.py            # DiceBCELoss, FocalLoss, TverskyLoss
│   │   ├── boundary.py          # Continuous SDF BoundaryDistLoss
│   │   ├── structure.py         # Soft-skeleton & CLDiceLoss
│   │   └── composite.py         # Compositional SegmentationLoss with warmup
│   ├── models/                  # Architecture builder & DAG compiler
│   │   └── model.py             # Declarative DAG compiler & auto-padding forward
│   ├── nn/                      # Reusable neural network building blocks
│   │   └── modules/
│   │       └── block.py         # CBA, Down, LKR, Ctx, Fuse, Agg, SegHead
│   └── utils/                   # Domain-segregated utility modules
│       ├── checkpoint.py        # Checkpoint serialization & loading
│       ├── ema.py               # Exponential Moving Average (ModelEMA)
│       ├── metrics.py           # IoU, Dice, Panoptic Quality metrics
│       └── rle.py               # Run-Length Encoding serialization
│
├── tests/                       # Automated test suite
│   ├── test_models.py           # Model instantiation & shape regression tests
│   └── test_losses.py           # Numerical stability & gradient tests
│
├── cli.py                       # Root entrypoint redirecting to soar.cli:main
├── pyproject.toml               # Modern build configuration (pip install -e .)
├── requirements.txt             # Framework dependencies
└── README.md                    # Framework documentation
```

## Quick Start

### Training

Train with gradient accumulation:

```bash
python cli.py train \
    --data /path/to/dataset \
    --model cfg/models/soar_medium1.yaml \
    --img-size 2048 2048 \
    --accumulate-grad-batches 4 \
    --epochs 100 \
    --lr 1e-4 \
    --amp \
    --device cuda
```

### Validation

Stream-validate without host memory explosion:

```bash
python cli.py val \
    --weights checkpoints/best.pt \
    --data /path/to/dataset \
    --model cfg/models/soar_medium1.yaml \
    --save-dir val_visualizations
```

### Prediction

Run full-resolution inference:

```bash
python cli.py predict \
    --weights checkpoints/best.pt \
    --data /path/to/test/dataset \
    --model cfg/models/soar_medium1.yaml \
    --output-dir predictions \
    --output-format image
```

## Model Architectures & Scales

SOAR1 scales its capacity through **Resolution-Aware Compound Scaling**, allocating depth and width according to spatial activation cost:

| Variant | Config File | Scale ID | Backbone Channels (P2 / P3 / P4 / P5) | Parameters |
| :--- | :--- | :---: | :---: | :---: |
| **SOAR1-Nano** | [`soar_nano1.yaml`](file:///e:/GithubProjects/OPT-HQ-Net/cfg/models/soar_nano1.yaml) | `n` | 32 / 64 / 128 / 256 | 0.94M |
| **SOAR1-Small** | [`soar_small1.yaml`](file:///e:/GithubProjects/OPT-HQ-Net/cfg/models/soar_small1.yaml) | `s` | 48 / 96 / 192 / 384 | 2.89M |
| **SOAR1-Medium** | [`soar_medium1.yaml`](file:///e:/GithubProjects/OPT-HQ-Net/cfg/models/soar_medium1.yaml) | `m` | 64 / 128 / 256 / 512 | 6.80M |
| **SOAR1-Large** | [`soar_large1.yaml`](file:///e:/GithubProjects/OPT-HQ-Net/cfg/models/soar_large1.yaml) | `l` | 64 / 160 / 320 / 640 | 13.47M |
| **SOAR1-XLarge** | [`soar_xlarge1.yaml`](file:///e:/GithubProjects/OPT-HQ-Net/cfg/models/soar_xlarge1.yaml) | `x` | 80 / 192 / 384 / 768 | 22.64M |
