# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, Callable
import albumentations as A


class BasePreprocessor:
    """Base class for image preprocessing."""
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess image and return processed image and optional valid mask."""
        raise NotImplementedError


class Normalize01(BasePreprocessor):
    """Normalize image to [0, 1] range using percentile clipping."""
    
    def __init__(self, percentiles: Tuple[float, float] = (1.0, 99.0)):
        self.percentiles = percentiles
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        lo, hi = np.percentile(img, self.percentiles)
        denom = max(float(hi - lo), 1e-6)
        normalized = np.clip((img - lo) / denom, 0.0, 1.0).astype(np.float32)
        return normalized, None


class CLAHE(BasePreprocessor):
    """Contrast Limited Adaptive Histogram Equalization."""
    
    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(u8).astype(np.float32) / 255.0
        return enhanced, None


class Resize(BasePreprocessor):
    """Resize image to target size."""
    
    def __init__(self, size: Tuple[int, int] = (1024, 1024), interpolation: int = cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        resized = cv2.resize(img, self.size, interpolation=self.interpolation)
        return resized, None


class PadToSize(BasePreprocessor):
    """Pad image to target size while maintaining aspect ratio."""
    
    def __init__(self, size: Tuple[int, int] = (1024, 1024), mode: str = 'constant', value: float = 0.0):
        self.size = size
        self.mode = mode
        self.value = value
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
        
        return padded, valid_mask


class ComposePreprocess:
    """Compose multiple preprocessing steps."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        valid_mask = None
        for t in self.transforms:
            img, mask = t(img)
            if mask is not None:
                if valid_mask is None:
                    valid_mask = mask
                else:
                    valid_mask = valid_mask * mask
        return img, valid_mask


def get_training_preprocessor(img_size: Tuple[int, int] = (1024, 1024)) -> ComposePreprocess:
    """Get standard training preprocessing pipeline."""
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        PadToSize(size=img_size, mode='constant', value=0.0),
    ])


def get_validation_preprocessor(img_size: Tuple[int, int] = (1024, 1024)) -> ComposePreprocess:
    """Get validation preprocessing pipeline."""
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        PadToSize(size=img_size, mode='constant', value=0.0),
    ])


def get_inference_preprocessor(img_size: Tuple[int, int] = (1024, 1024)) -> ComposePreprocess:
    """Get inference preprocessing pipeline."""
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        PadToSize(size=img_size, mode='constant', value=0.0),
    ])
