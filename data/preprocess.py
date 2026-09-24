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


class LetterBox(BasePreprocessor):
    """Resize and pad image to target size while preserving aspect ratio (Ultralytics style).
    
    Handles cases where target size is larger or smaller than actual image size:
    - If target > actual: scales up and pads (letterboxing)
    - If target < actual: scales down and pads (letterboxing)
    - Preserves aspect ratio
    - Supports mod 32 alignment for stride compatibility
    - Supports rectangular inference (minimal padding)
    """
    
    def __init__(
        self,
        new_shape: Tuple[int, int] = (1024, 1024),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: float = 114.0,  # Ultralytics uses 114 for gray padding
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
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


def get_training_preprocessor(img_size: Tuple[int, int] = (1024, 1024), auto: bool = False) -> ComposePreprocess:
    """Get standard training preprocessing pipeline with LetterBox.
    
    Args:
        img_size: Target image size (height, width)
        auto: If True, use minimum rectangle with mod 32 alignment (more efficient)
    """
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        LetterBox(new_shape=img_size, scaleup=True, center=True, auto=auto, padding_value=114.0),
    ])


def get_validation_preprocessor(img_size: Tuple[int, int] = (1024, 1024), auto: bool = True) -> ComposePreprocess:
    """Get validation preprocessing pipeline with LetterBox (no scaleup for better mAP).
    
    Args:
        img_size: Target image size (height, width)
        auto: If True, use minimum rectangle with mod 32 alignment (more efficient, default)
    """
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        LetterBox(new_shape=img_size, scaleup=False, center=True, auto=auto, padding_value=114.0),
    ])


def get_inference_preprocessor(img_size: Tuple[int, int] = (1024, 1024), auto: bool = True) -> ComposePreprocess:
    """Get inference preprocessing pipeline with LetterBox.
    
    Args:
        img_size: Target image size (height, width)
        auto: If True, use minimum rectangle with mod 32 alignment (more efficient, default)
    """
    return ComposePreprocess([
        Normalize01(percentiles=(1.0, 99.0)),
        CLAHE(clip_limit=2.5, tile_grid_size=(8, 8)),
        LetterBox(new_shape=img_size, scaleup=True, center=True, auto=auto, padding_value=114.0),
    ])
