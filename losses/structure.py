# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseLoss


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological erosion."""
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological dilation."""
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological opening."""
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
    """Soft skeletonization for connectivity-aware loss."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(n_iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CLDiceLoss(BaseLoss):
    """
    ClDice loss for thin structure segmentation.
    
    Connectivity-aware loss using skeletonization.
    """
    
    def __init__(self, weight: float = 1.0, n_iter: int = 3, smooth: float = 1.0):
        super().__init__(weight)
        self.n_iter = n_iter
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        
        if valid_mask is not None:
            prob = prob * valid_mask
            target = target * valid_mask
        
        skel_pred = soft_skeletonize(prob, self.n_iter)
        skel_true = soft_skeletonize(target, self.n_iter)

        t_prec = (torch.sum(skel_pred * target) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        t_sens = (torch.sum(skel_true * prob) + self.smooth) / (torch.sum(skel_true) + self.smooth)

        cldice = 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + 1e-8)
        return self.weight * cldice


class SkeletonLoss(BaseLoss):
    """Skeleton loss for thin structure preservation."""
    
    def __init__(self, weight: float = 1.0, n_iter: int = 3):
        super().__init__(weight)
        self.n_iter = n_iter
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        
        if valid_mask is not None:
            prob = prob * valid_mask
            target = target * valid_mask
        
        skel_pred = soft_skeletonize(prob, self.n_iter)
        skel_true = soft_skeletonize(target, self.n_iter)
        
        loss = F.l1_loss(skel_pred, skel_true, reduction="none")
        return self.weight * self._apply_valid_mask(loss, valid_mask)
