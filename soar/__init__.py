from __future__ import annotations

"""
SOAR: Resolution-Preserving Scientific Segmentation Framework
============================================================
A dedicated deep learning segmentation library engineered for native high-resolution
imagery (2048x2048+) with extreme class imbalance and ultra-thin, curvilinear topological structures.
"""

__version__ = "1.0.0"

from .models.model import SegmentationModel
from .engine.trainer import BaseTrainer
from .engine.validator import BaseValidator
from .engine.predictor import BasePredictor
from .losses.composite import SegmentationLoss
from .data.dataset import SegmentationDataset, collate_fn
from .data.config import PreprocessConfig
from .utils.checkpoint import save_checkpoint, load_checkpoint
from .utils.ema import ModelEMA
from .utils.metrics import compute_iou, compute_dice, compute_panoptic_quality
from .utils.rle import binary_mask_to_rle, rle_to_binary_mask

__all__ = [
    "__version__",
    "SegmentationModel",
    "BaseTrainer",
    "BaseValidator",
    "BasePredictor",
    "SegmentationLoss",
    "SegmentationDataset",
    "collate_fn",
    "PreprocessConfig",
    "save_checkpoint",
    "load_checkpoint",
    "ModelEMA",
    "compute_iou",
    "compute_dice",
    "compute_panoptic_quality",
    "binary_mask_to_rle",
    "rle_to_binary_mask",
]
