# High-Resolution Segmentation Framework

A general-purpose, high-resolution semantic segmentation framework inspired by Ultralytics YOLO architecture. Designed for flexibility and ease of use across various segmentation tasks.

## Features

- **Modular Architecture**: Separated into reusable components (engine, models, nn, data, cfg)
- **YAML-Based Model Configuration**: Define model architectures in YAML files for easy experimentation
- **Flexible Data Pipeline**: General-purpose dataset loader supporting multiple formats (COCO JSON, mask directories)
- **Comprehensive Augmentation**: Built-in data augmentation pipeline following Ultralytics patterns
- **Advanced Loss Functions**: Combines BCE, Dice, ClDice (for thin structures), and boundary losses
- **Training/Validation/Prediction Engines**: Separate engines for each phase
- **Distributed Training**: Support for multi-GPU training with DDP
- **High-Resolution Support**: Optimized for large image segmentation (e.g., 2048x2048)

## Project Structure

```
OPT-HQ-Net/
├── cfg/                      # Configuration files
│   ├── default.yaml         # Default training configuration
│   └── models/              # Model architecture YAMLs
│       ├── seg_hrnet.yaml   # High-resolution segmentation model
│       └── seg_light.yaml   # Lightweight segmentation model
├── data/                    # Data processing modules
│   ├── augment.py          # Data augmentation pipeline
│   ├── dataset.py          # General-purpose dataset loader
│   └── preprocess.py       # Image preprocessing
├── engine/                  # Training/validation/prediction engines
│   ├── trainer.py          # Training engine
│   ├── validator.py        # Validation engine
│   └── predictor.py        # Prediction engine
├── models/                  # Model definitions
│   └── model.py            # YAML-based model parser and builder
├── nn/                      # Neural network building blocks
│   └── modules/            # Reusable modules
│       └── block.py        # Conv, Bottleneck, C3k2, SPPF, etc.
├── cli.py                   # Command-line interface
├── losses.py               # Segmentation loss functions
├── utils.py                # Utility functions (EMA, checkpointing, metrics)
└── requirements.txt         # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train a segmentation model with default settings:

```bash
python cli.py train --data /path/to/dataset --model cfg/models/seg_hrnet.yaml
```

With custom parameters:

```bash
python cli.py train \
    --data /path/to/dataset \
    --model cfg/models/seg_hrnet.yaml \
    --img-size 1024 1024 \
    --batch-size 2 \
    --epochs 100 \
    --lr 1e-4 \
    --device cuda \
    --checkpoint-dir checkpoints
```

### Validation

Validate a trained model:

```bash
python cli.py val \
    --weights checkpoints/best.pt \
    --data /path/to/dataset \
    --model cfg/models/seg_hrnet.yaml \
    --save-dir val_visualizations
```

### Prediction

Run inference on test images:

```bash
python cli.py predict \
    --weights checkpoints/best.pt \
    --data /path/to/test/dataset \
    --model cfg/models/seg_hrnet.yaml \
    --output-dir predictions \
    --output-format image
```

For competition submission (RLE format):

```bash
python cli.py predict \
    --weights checkpoints/best.pt \
    --data /path/to/test/dataset \
    --model cfg/models/seg_hrnet.yaml \
    --output-dir predictions \
    --output-format rle
```

## Dataset Format

The framework supports multiple dataset formats:

### COCO JSON Format

```
dataset/
├── train/
│   ├── train_images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── annotations.json
└── test/
    └── test_images/
        ├── test1.jpg
        └── test2.jpg
```

### Mask Directory Format

```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── masks/
│       ├── image1.png
│       └── image2.png
└── test/
    └── images/
        ├── test1.jpg
        └── test2.jpg
```

## Model Configuration

Models are defined in YAML files in `cfg/models/`. Example structure:

```yaml
nc: 1  # number of classes
channels: 3  # input channels

# Backbone - Feature extraction
backbone:
  # [from, n, module, args]
  [-1, 1, Conv, [32, 3, 2]]      # downsample
  [-1, 1, Conv, [64, 3, 2]]      # downsample (P2)
  [-1, 1, SPPF, [64, 5]]         # spatial pyramid pooling

# Neck - Feature fusion
neck:
  [-1, 1, Upsample, [2, 'nearest']]
  [[-1, 1], 1, Conv, [32, 1, 1]]

# Head - Segmentation head
head:
  [-1, 1, Conv, [16, 1, 1]]
  [-1, 1, nn.Conv2d, [1, 1, 1]]  # final output
```

### Available Modules

- `Conv`: Standard convolution with GroupNorm and SiLU
- `Bottleneck`: Standard bottleneck block
- `C3k2`: CSP bottleneck with k=2
- `SPPF`: Spatial Pyramid Pooling - Fast
- `Upsample`: Upsampling layer
- `Concat`: Concatenation layer

## Custom Models

Create custom model architectures by modifying YAML files or adding new ones:

1. Copy an existing model YAML: `cfg/models/seg_hrnet.yaml`
2. Modify the backbone, neck, and head sections
3. Train with: `python cli.py train --model cfg/models/your_model.yaml`

## Advanced Features

### Distributed Training

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 cli.py train --data /path/to/dataset
```

### Automatic Mixed Precision

```bash
python cli.py train --data /path/to/dataset --amp
```

### Exponential Moving Average

```bash
python cli.py train --data /path/to/dataset --ema
```

### Resume Training

```bash
python cli.py train --data /path/to/dataset --resume checkpoints/last.pt
```

## Loss Functions

The framework uses a composite loss function:

- **BCE Loss**: Binary cross-entropy with positive weighting
- **Dice Loss**: Dice coefficient for overlap optimization
- **ClDice Loss**: Connectivity-aware loss for thin structures (with warmup)
- **Boundary Loss**: Edge-aware loss using Sobel filters

Configure loss weights in `cfg/default.yaml`:

```yaml
loss:
  w_bce: 1.0
  w_dice: 1.0
  w_cldice_target: 0.2
  w_boundary: 0.2
```

## Data Augmentation

Built-in augmentations include:

- Random rotation (0-360 degrees)
- Random horizontal/vertical flips
- Random brightness/contrast adjustment
- Random Gaussian blur

Configure in `data/augment.py`.

## Preprocessing

Standard preprocessing pipeline:

1. Percentile-based normalization (1-99%)
2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Padding to target size

## Migration from Original Code

The original solar filament-specific code has been refactored into a general-purpose framework:

- `main.py` → `cli.py` (new CLI interface)
- `model.py` → `models/model.py` + `nn/modules/block.py` (YAML-based)
- `dataset.py` → `data/dataset.py` (generalized)
- `preprocessing.py` → `data/preprocess.py` (generalized)
- `losses.py` → `losses.py` (renamed MicroFilNetLoss to SegmentationLoss)
- `utils.py` → `utils.py` (generalized utilities)

Old files are preserved for reference but should use the new framework for new projects.

## Citation

If you use this framework in your research, please cite:

```
@software{opt_hq_net,
  title={High-Resolution Segmentation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/OPT-HQ-Net}
}
```

## License

AGPL-3.0 License - See LICENSE file for details.

## Acknowledgments

- Inspired by Ultralytics YOLO architecture
- Original solar filament segmentation work
