# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import cv2
import numpy as np


@dataclass
class CanonicalConfig:
    """Canonical conversion settings (framework-owned).
    
    These are safe tensor preparation operations that the framework
    handles automatically: dtype conversion, channel ordering, tensorization.
    """
    dtype: str = "float32"  # Target dtype
    normalize_255: bool = False  # Whether to divide by 255 (only for uint8 RGB)
    channel_order: str = "chw"  # "chw" or "hwc"


@dataclass
class GeometricConfig:
    """Geometric transform settings (user-controlled).
    
    These can alter spatial information and should be explicitly enabled.
    """
    enabled: bool = False  # Whether to apply geometric transforms
    resize: bool = False  # Whether to resize
    target_size: Optional[Tuple[int, int]] = None  # (height, width)
    keep_native_resolution: bool = True  # If True, preserve original resolution
    letterbox: bool = False  # Whether to use letterboxing (aspect-ratio-preserving)
    auto: bool = False  # Use minimum rectangle with mod 32 alignment
    scaleup: bool = True  # Allow scaling up
    center: bool = True  # Center the padded image
    stride: int = 32  # Stride for mod alignment
    flip_horizontal: bool = False  # Random horizontal flip
    flip_vertical: bool = False  # Random vertical flip
    rotate: bool = False  # Random rotation
    rotation_degrees: float = 0.0  # Max rotation degrees


@dataclass
class AppearanceConfig:
    """Appearance transform settings (user-controlled).
    
    These can alter semantic/physical image information and should be
    explicitly enabled by the user.
    """
    enabled: bool = False  # Whether to apply appearance transforms
    clahe: bool = False  # Contrast Limited Adaptive Histogram Equalization
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    normalize_percentile: bool = False  # Percentile-based normalization
    normalize_percentiles: Tuple[float, float] = (1.0, 99.0)
    normalize_zscore: bool = False  # Z-score normalization
    normalize_minmax: bool = False  # Min-max normalization
    brightness: float = 0.0  # Brightness adjustment (-1 to 1)
    contrast: float = 0.0  # Contrast adjustment (-1 to 1)
    saturation: float = 0.0  # Saturation adjustment (-1 to 1)
    hue: float = 0.0  # Hue adjustment (-1 to 1)
    noise: float = 0.0  # Gaussian noise std


@dataclass
class PreprocessConfig:
    """Complete preprocessing configuration.
    
    Philosophy:
    - Framework owns canonical conversion (tensorization, dtype)
    - User controls geometric transforms (resize, letterbox)
    - User controls appearance transforms (CLAHE, normalization)
    - Nothing is silently enabled
    - Training/validation have separate pipelines
    """
    canonical: CanonicalConfig = CanonicalConfig()
    geometric: GeometricConfig = GeometricConfig()
    appearance: AppearanceConfig = AppearanceConfig()
    
    @classmethod
    def minimal(cls) -> "PreprocessConfig":
        """Minimal preprocessing: only canonical conversion."""
        return cls(
            canonical=CanonicalConfig(),
            geometric=GeometricConfig(enabled=False),
            appearance=AppearanceConfig(enabled=False),
        )
    
    @classmethod
    def standard_training(cls, img_size: Tuple[int, int] = (1024, 1024)) -> "PreprocessConfig":
        """Standard training with geometric transforms but no appearance transforms."""
        return cls(
            canonical=CanonicalConfig(),
            geometric=GeometricConfig(
                enabled=True,
                resize=True,
                target_size=img_size,
                keep_native_resolution=False,
                letterbox=True,
                auto=False,
                scaleup=True,
                center=True,
                flip_horizontal=True,
            ),
            appearance=AppearanceConfig(enabled=False),
        )
    
    @classmethod
    def standard_validation(cls, img_size: Tuple[int, int] = (1024, 1024)) -> "PreprocessConfig":
        """Standard validation: no augmentation, only geometric if needed."""
        return cls(
            canonical=CanonicalConfig(),
            geometric=GeometricConfig(
                enabled=True,
                resize=True,
                target_size=img_size,
                keep_native_resolution=False,
                letterbox=True,
                auto=True,  # Efficient rectangular inference
                scaleup=False,  # Only scale down for better mAP
                center=True,
                flip_horizontal=False,
                flip_vertical=False,
                rotate=False,
            ),
            appearance=AppearanceConfig(enabled=False),
        )
    
    @classmethod
    def native_resolution(cls) -> "PreprocessConfig":
        """Keep native resolution - no geometric transforms."""
        return cls(
            canonical=CanonicalConfig(),
            geometric=GeometricConfig(
                enabled=False,
                keep_native_resolution=True,
            ),
            appearance=AppearanceConfig(enabled=False),
        )
