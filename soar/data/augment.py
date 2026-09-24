from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


class BaseAugmentation:
    """Base class for data augmentation pipelines."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying this augmentation.
        """
        self.p = p

    def __call__(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation to image and optionally mask."""
        if random.random() < self.p:
            return self._apply(img, mask)
        return img, mask

    def _apply(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """To be implemented by subclasses."""
        raise NotImplementedError


class RandomRotate(BaseAugmentation):
    """Random affine rotation preserving structural continuity."""

    def __init__(self, p: float = 0.5, angles: Tuple[float, float] = (0.0, 360.0)):
        super().__init__(p)
        self.angles = angles

    def _apply(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        angle = random.uniform(*self.angles)
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        if img.ndim == 2:
            img_out = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        elif img.ndim == 3:
            # Assumes HWC format from dataset.py
            img_out = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        mask_out = None
        if mask is not None:
            # Critical: Use INTER_NEAREST for masks to avoid interpolating labels
            mask_out = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)

        return img_out, mask_out


class RandomFlip(BaseAugmentation):
    """Random horizontal and vertical reflection."""

    def __init__(self, p: float = 0.5, horizontal: bool = True, vertical: bool = True):
        super().__init__(p)
        self.horizontal = horizontal
        self.vertical = vertical

    def _apply(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img_out = img.copy()
        mask_out = mask.copy() if mask is not None else None

        # Horizontal flip (axis 1 for HWC and HW)
        if self.horizontal and random.random() < 0.5:
            img_out = np.flip(img_out, axis=1)
            if mask_out is not None:
                mask_out = np.flip(mask_out, axis=1)

        # Vertical flip (axis 0 for HWC and HW)
        if self.vertical and random.random() < 0.5:
            img_out = np.flip(img_out, axis=0)
            if mask_out is not None:
                mask_out = np.flip(mask_out, axis=0)

        return np.ascontiguousarray(img_out), (
            np.ascontiguousarray(mask_out) if mask_out is not None else None
        )


class RandomBrightnessContrast(BaseAugmentation):
    """Random brightness scaling and contrast adjustment."""

    def __init__(
        self,
        p: float = 0.5,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ):
        super().__init__(p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def _apply(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)

        img_out = img.astype(np.float32) * brightness
        mean_val = float(img_out.mean())
        img_out = (img_out - mean_val) * contrast + mean_val
        img_out = np.clip(img_out, 0.0, 1.0).astype(img.dtype)

        return img_out, mask


class RandomGaussianBlur(BaseAugmentation):
    """Random Gaussian blur with guaranteed positive, odd kernel sizes."""

    def __init__(
        self,
        p: float = 0.5,
        kernel_size_range: Tuple[int, int] = (3, 7),
        sigma_range: Tuple[float, float] = (0.1, 2.0),
    ):
        super().__init__(p)
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range

    def _apply(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        k_min, k_max = self.kernel_size_range
        kernel_size = random.randint(k_min, k_max)

        # Enforce kernel_size to be a strictly positive odd integer
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = float(random.uniform(*self.sigma_range))

        if img.ndim == 2:
            img_out = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        elif img.ndim == 3:
            # OpenCV GaussianBlur supports HWC directly (e.g., 3-channel RGB or multi-channel)
            img_out = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        else:
            raise ValueError(f"Unexpected image dimensions for blur: {img.shape}")

        return img_out, mask


class Compose:
    """Sequential composition of multiple augmentation steps."""

    def __init__(self, transforms: List[BaseAugmentation]):
        self.transforms = transforms

    def __call__(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def get_training_augmentation() -> Compose:
    """Standard training augmentation pipeline."""
    return Compose([
        RandomRotate(p=0.5, angles=(0.0, 360.0)),
        RandomFlip(p=0.5, horizontal=True, vertical=True),
        RandomBrightnessContrast(p=0.3, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)),
        RandomGaussianBlur(p=0.2, kernel_size_range=(3, 7), sigma_range=(0.1, 1.5)),
    ])


def get_validation_augmentation() -> Compose:
    """Validation pipeline (identity pass-through)."""
    return Compose([])