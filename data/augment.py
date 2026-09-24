# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, Callable
import random


class BaseAugmentation:
    """Base class for data augmentation pipelines."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying this augmentation
        """
        self.p = p
    
    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation to image and optionally mask."""
        if random.random() < self.p:
            return self._apply(img, mask)
        return img, mask
    
    def _apply(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """To be implemented by subclasses."""
        raise NotImplementedError


class RandomRotate(BaseAugmentation):
    """Random rotation augmentation."""
    
    def __init__(self, p: float = 0.5, angles: Tuple[float, float] = (0.0, 360.0)):
        super().__init__(p)
        self.angles = angles
    
    def _apply(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        angle = random.uniform(*self.angles)
        h, w = img.shape[-2:]
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if img.ndim == 3:
            img_out = np.stack([cv2.warpAffine(img[ch], rot_mat, (w, h), flags=cv2.INTER_LINEAR) for ch in range(img.shape[0])])
        else:
            img_out = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        
        if mask is not None:
            mask_out = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
            return img_out, mask_out
        
        return img_out, None


class RandomFlip(BaseAugmentation):
    """Random horizontal/vertical flip augmentation."""
    
    def __init__(self, p: float = 0.5, horizontal: bool = True, vertical: bool = True):
        super().__init__(p)
        self.horizontal = horizontal
        self.vertical = vertical
    
    def _apply(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img_out = img.copy()
        mask_out = mask.copy() if mask is not None else None
        
        if self.horizontal and random.random() < 0.5:
            img_out = img_out[..., ::-1]
            if mask_out is not None:
                mask_out = mask_out[..., ::-1]
        
        if self.vertical and random.random() < 0.5:
            img_out = img_out[..., ::-1, :]
            if mask_out is not None:
                mask_out = mask_out[..., ::-1, :]
        
        return img_out, mask_out


class RandomBrightnessContrast(BaseAugmentation):
    """Random brightness and contrast adjustment."""
    
    def __init__(self, p: float = 0.5, brightness_range: Tuple[float, float] = (0.8, 1.2), 
                 contrast_range: Tuple[float, float] = (0.8, 1.2)):
        super().__init__(p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def _apply(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        
        img_out = img.astype(np.float32)
        img_out = img_out * brightness
        img_out = (img_out - img_out.mean()) * contrast + img_out.mean()
        img_out = np.clip(img_out, 0.0, 1.0)
        
        return img_out, mask


class RandomGaussianBlur(BaseAugmentation):
    """Random Gaussian blur augmentation."""
    
    def __init__(self, p: float = 0.5, kernel_size_range: Tuple[int, int] = (3, 7), sigma_range: Tuple[float, float] = (0.1, 2.0)):
        super().__init__(p)
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
    
    def _apply(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        kernel_size = random.randint(*self.kernel_size_range)
        sigma = random.uniform(*self.sigma_range)
        
        if img.ndim == 3:
            img_out = np.stack([cv2.GaussianBlur(img[ch], (kernel_size, kernel_size), sigma) for ch in range(img.shape[0])])
        else:
            img_out = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        
        return img_out, mask


class Compose:
    """Compose multiple augmentations."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def get_training_augmentation() -> Compose:
    """Get standard training augmentation pipeline."""
    return Compose([
        RandomRotate(p=0.5, angles=(0.0, 360.0)),
        RandomFlip(p=0.5, horizontal=True, vertical=True),
        RandomBrightnessContrast(p=0.3),
        RandomGaussianBlur(p=0.2),
    ])


def get_validation_augmentation() -> Compose:
    """Get validation augmentation (typically identity)."""
    return Compose([])
