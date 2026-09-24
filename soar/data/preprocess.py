from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
import albumentations as A

from .preprocess_config import PreprocessConfig, CanonicalConfig, GeometricConfig, AppearanceConfig


class BasePreprocessor:
    """Base class for image preprocessing."""
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Preprocess image and return processed image, valid mask, and metadata."""
        raise NotImplementedError


class Normalize01(BasePreprocessor):
    """Normalize image to [0, 1] range using percentile clipping (appearance transform)."""
    
    def __init__(self, percentiles: Tuple[float, float] = (1.0, 99.0)):
        self.percentiles = percentiles
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        lo, hi = np.percentile(img, self.percentiles)
        denom = max(float(hi - lo), 1e-6)
        normalized = np.clip((img - lo) / denom, 0.0, 1.0).astype(np.float32)
        meta = {'normalization': 'percentile', 'percentiles': self.percentiles}
        return normalized, None, meta


class NormalizeMinMax(BasePreprocessor):
    """Min-max normalization to [0, 1] range (appearance transform)."""
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        min_val = img.min()
        max_val = img.max()
        denom = max(float(max_val - min_val), 1e-6)
        normalized = ((img - min_val) / denom).astype(np.float32)
        meta = {'normalization': 'minmax', 'min': float(min_val), 'max': float(max_val)}
        return normalized, None, meta


class NormalizeZScore(BasePreprocessor):
    """Z-score normalization (mean=0, std=1) (appearance transform)."""
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        mean = img.mean()
        std = img.std()
        denom = max(float(std), 1e-6)
        normalized = ((img - mean) / denom).astype(np.float32)
        meta = {'normalization': 'zscore', 'mean': float(mean), 'std': float(std)}
        return normalized, None, meta


class CLAHE(BasePreprocessor):
    """Contrast Limited Adaptive Histogram Equalization (appearance transform)."""
    
    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(u8).astype(np.float32) / 255.0
        meta = {'clahe': True, 'clip_limit': self.clip_limit, 'tile_grid_size': self.tile_grid_size}
        return enhanced, None, meta


class Resize(BasePreprocessor):
    """Resize image to target size (geometric transform)."""
    
    def __init__(self, size: Tuple[int, int] = (1024, 1024), interpolation: int = cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        resized = cv2.resize(img, self.size, interpolation=self.interpolation)
        meta = {'resize': True, 'target_size': self.size, 'original_shape': img.shape[:2]}
        return resized, None, meta


class PadToSize(BasePreprocessor):
    """Pad image to target size while maintaining aspect ratio (geometric transform)."""
    
    def __init__(self, size: Tuple[int, int] = (1024, 1024), mode: str = 'constant', value: float = 0.0):
        self.size = size
        self.mode = mode
        self.value = value
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        h, w = img.shape[-2:]
        target_h, target_w = self.size
        
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        if img.ndim == 3:
            padding = ((0, 0), (0, pad_w), (0, pad_h))[:img.ndim]
        else:
            padding = ((0, pad_w), (0, pad_h))
        
        padded = np.pad(img, padding, mode=self.mode, constant_values=self.value)
        valid_mask = np.ones_like(padded)
        if pad_h > 0:
            valid_mask[..., -pad_h:] = 0
        if pad_w > 0:
            valid_mask[..., -pad_w:, :] = 0
        
        meta = {'pad': True, 'target_size': self.size, 'padding': (pad_h, pad_w)}
        return padded, valid_mask, meta


class LetterBox(BasePreprocessor):
    """Resize and pad image to target size while preserving aspect ratio.
    
    Handles cases where target size is larger or smaller than actual image size:
    - If target > actual: scales up and pads (letterboxing)
    - If target < actual: scales down and pads (letterboxing)
    - Preserves aspect ratio
    - Supports mod 32 alignment for stride compatibility
    - Supports rectangular inference (minimal padding)
    
    For scientific segmentation (normalized float data):
    - Uses padding_value=0.0 (background) instead of 114.0 (RGB gray)
    - Returns metadata for unpadding during inference
    """
    
    def __init__(
        self,
        new_shape: Tuple[int, int] = (1024, 1024),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: float = 0.0,  # 0.0 for normalized float data (background)
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply letterbox transform.
        
        Returns:
            padded: Padded image
            valid_mask: Mask indicating valid regions (1) vs padding (0)
            meta: Transform metadata for unpadding (ratio, padding, original shape)
        """
        # Handle both (H, W) and (H, W, C) shapes
        if img.ndim == 2:
            img = img[..., None]
        
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        
        # Scale ratio (new / old) - constrain to long edge
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if self.auto:  # minimum rectangle - mod 32 alignment
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:  # stretch - no padding
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
        
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
        
        # Pad
        if img.ndim == 2:
            img = img[..., None]
        
        h, w, c = img.shape
        padded = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(self.padding_value,) * c if c > 1 else self.padding_value
        )
        
        # Create valid mask (1 for real image, 0 for padding)
        valid_mask = np.ones((padded.shape[0], padded.shape[1]), dtype=np.float32)
        if top > 0:
            valid_mask[:top, :] = 0
        if bottom > 0:
            valid_mask[-bottom:, :] = 0
        if left > 0:
            valid_mask[:, :left] = 0
        if right > 0:
            valid_mask[:, -right:] = 0
        
        # Store transform metadata for unpadding
        meta = {
            'ratio': ratio,  # (width_ratio, height_ratio)
            'padding': (top, bottom, left, right),
            'original_shape': shape,  # (height, width)
            'new_shape': new_shape,  # (height, width)
            'new_unpad': new_unpad,  # (width, height) after resize
        }
        
        return padded, valid_mask, meta


class LetterBoxMask(BasePreprocessor):
    """Resize and pad mask to target size using INTER_NEAREST interpolation (geometric transform).
    
    Critical for segmentation: masks must use nearest-neighbor interpolation to avoid
    introducing fractional values at boundaries (e.g., 0.15, 0.65) which corrupt
    edge-loss calculations and single-pixel-wide filament boundaries.
    
    Uses same transform parameters as LetterBox for consistency.
    """
    
    def __init__(
        self,
        new_shape: Tuple[int, int] = (1024, 1024),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 0,  # 0 for mask (background/ignore)
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
    
    def __call__(self, mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply letterbox transform to mask with INTER_NEAREST.
        
        Returns:
            padded: Padded mask
            valid_mask: Mask indicating valid regions (1) vs padding (0)
            meta: Transform metadata for unpadding
        """
        # Handle both (H, W) and (H, W, C) shapes
        if mask.ndim == 2:
            mask = mask[..., None]
        
        shape = mask.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        
        # Scale ratio (new / old) - constrain to long edge
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        if self.center:
            dw /= 2
            dh /= 2
        
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
        # Resize with INTER_NEAREST (critical for masks)
        if shape[::-1] != new_unpad:
            mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_NEAREST)
        
        # Pad
        if mask.ndim == 2:
            mask = mask[..., None]
        
        h, w, c = mask.shape
        padded = cv2.copyMakeBorder(
            mask, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(self.padding_value,) * c if c > 1 else self.padding_value
        )
        
        # Create valid mask
        valid_mask = np.ones((padded.shape[0], padded.shape[1]), dtype=np.float32)
        if top > 0:
            valid_mask[:top, :] = 0
        if bottom > 0:
            valid_mask[-bottom:, :] = 0
        if left > 0:
            valid_mask[:, :left] = 0
        if right > 0:
            valid_mask[:, -right:] = 0
        
        # Store transform metadata
        meta = {
            'ratio': ratio,
            'padding': (top, bottom, left, right),
            'original_shape': shape,
            'new_shape': new_shape,
            'new_unpad': new_unpad,
        }
        
        return padded, valid_mask, meta


class ComposePreprocess:
    """Compose multiple preprocessing steps."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """Apply composed transforms.
        
        Returns:
            img: Processed image
            valid_mask: Mask indicating valid regions
            meta: Combined transform metadata
        """
        valid_mask = None
        meta = {}
        for t in self.transforms:
            result = t(img)
            if len(result) == 3:
                img, mask, step_meta = result
                if step_meta:
                    meta.update(step_meta)
            else:
                img, mask = result
            if mask is not None:
                if valid_mask is None:
                    valid_mask = mask
                else:
                    valid_mask = valid_mask * mask
        return img, valid_mask, meta


def build_preprocessor_from_config(config: PreprocessConfig) -> ComposePreprocess:
    """Build preprocessing pipeline from configuration.
    
    Args:
        config: PreprocessConfig with canonical, geometric, and appearance settings
    
    Returns:
        ComposePreprocess pipeline
    """
    transforms = []
    
    # Appearance transforms (user-controlled)
    if config.appearance.enabled:
        if config.appearance.normalize_percentile:
            transforms.append(Normalize01(percentiles=config.appearance.normalize_percentiles))
        elif config.appearance.normalize_minmax:
            transforms.append(NormalizeMinMax())
        elif config.appearance.normalize_zscore:
            transforms.append(NormalizeZScore())
        
        if config.appearance.clahe:
            transforms.append(CLAHE(
                clip_limit=config.appearance.clahe_clip_limit,
                tile_grid_size=config.appearance.clahe_tile_grid_size,
            ))
    
    # Geometric transforms (user-controlled)
    if config.geometric.enabled:
        if config.geometric.letterbox and config.geometric.target_size:
            transforms.append(LetterBox(
                new_shape=config.geometric.target_size,
                auto=config.geometric.auto,
                scaleup=config.geometric.scaleup,
                center=config.geometric.center,
                stride=config.geometric.stride,
                padding_value=0.0,
            ))
        elif config.geometric.resize and config.geometric.target_size:
            transforms.append(Resize(size=config.geometric.target_size))
    
    return ComposePreprocess(transforms)


def get_training_preprocessor(img_size: Tuple[int, int] = (1024, 1024), auto: bool = False) -> ComposePreprocess:
    """Get standard training preprocessing pipeline with LetterBox.
    
    DEPRECATED: Use build_preprocessor_from_config with PreprocessConfig.standard_training()
    
    Args:
        img_size: Target image size (height, width)
        auto: If True, use minimum rectangle with mod 32 alignment (more efficient)
    """
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        LetterBox(new_shape=img_size, scaleup=True, center=True, auto=auto, padding_value=0.0),
    ])


def get_validation_preprocessor(img_size: Tuple[int, int] = (1024, 1024), auto: bool = True) -> ComposePreprocess:
    """Get validation preprocessing pipeline with LetterBox (no scaleup for better mAP).
    
    DEPRECATED: Use build_preprocessor_from_config with PreprocessConfig.standard_validation()
    
    Args:
        img_size: Target image size (height, width)
        auto: If True, use minimum rectangle with mod 32 alignment (more efficient, default)
    """
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        LetterBox(new_shape=img_size, scaleup=False, center=True, auto=auto, padding_value=0.0),
    ])


def get_inference_preprocessor(img_size: Tuple[int, int] = (1024, 1024), auto: bool = True) -> ComposePreprocess:
    """Get inference preprocessing pipeline with LetterBox.
    
    DEPRECATED: Use build_preprocessor_from_config with PreprocessConfig.standard_validation()
    
    Args:
        img_size: Target image size (height, width)
        auto: If True, use minimum rectangle with mod 32 alignment (more efficient, default)
    """
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        LetterBox(new_shape=img_size, scaleup=True, center=True, auto=auto, padding_value=0.0),
    ])


def get_mask_preprocessor(img_size: Tuple[int, int] = (1024, 1024), auto: bool = False, scaleup: bool = True) -> LetterBoxMask:
    """Get mask preprocessor with INTER_NEAREST interpolation.
    
    Args:
        img_size: Target image size (height, width)
        auto: If True, use minimum rectangle with mod 32 alignment
        scaleup: If True, allow scaling up
    """
    return LetterBoxMask(
        new_shape=img_size,
        scaleup=scaleup,
        center=True,
        auto=auto,
        padding_value=0,
    )
