# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseLoss


class BCELoss(BaseLoss):
    """Binary Cross-Entropy loss for segmentation."""
    
    def __init__(self, weight: float = 1.0, pos_weight: float = 1.0):
        super().__init__(weight)
        self.pos_weight = pos_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=pred.device, dtype=pred.dtype)
        loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw, reduction="none")
        return self.weight * self._apply_valid_mask(loss, valid_mask)


class DiceLoss(BaseLoss):
    """Dice loss for segmentation."""
    
    def __init__(self, weight: float = 1.0, smooth: float = 1.0):
        super().__init__(weight)
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
        
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()
        
        return self.weight * loss


class FocalLoss(BaseLoss):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, weight: float = 1.0, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(weight)
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        
        pt = prob * target + (1 - prob) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = self.alpha * focal_weight * bce
        return self.weight * self._apply_valid_mask(loss, valid_mask)


class TverskyLoss(BaseLoss):
    """Tversky loss for controlling FP/FN tradeoff."""
    
    def __init__(self, weight: float = 1.0, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__(weight)
        self.alpha = alpha
        self.beta = beta
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
        
        tp = (prob * target).sum(dim=(1, 2, 3))
        fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
        fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky.mean()
        
        return self.weight * loss


class FocalTverskyLoss(BaseLoss):
    """Focal Tversky loss for difficult, imbalanced segmentation."""
    
    def __init__(self, weight: float = 1.0, alpha: float = 0.5, beta: float = 0.5, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__(weight)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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
        
        tp = (prob * target).sum(dim=(1, 2, 3))
        fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
        fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        loss = focal_tversky.mean()
        
        return self.weight * loss


class DiceBCELoss(BaseLoss):
    """Combined Dice + BCE loss (strong generic baseline)."""
    
    def __init__(self, weight: float = 1.0, dice_weight: float = 1.0, bce_weight: float = 1.0, 
                 bce_pos_weight: float = 1.0, smooth: float = 1.0):
        super().__init__(weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_pos_weight = bce_pos_weight
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # BCE component
        pw = torch.tensor([self.bce_pos_weight], device=pred.device, dtype=pred.dtype)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw, reduction="none")
        bce_loss = self._apply_valid_mask(bce, valid_mask)
        
        # Dice component
        prob = torch.sigmoid(pred)
        if valid_mask is not None:
            prob = prob * valid_mask
            target = target * valid_mask
        
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        # Combine
        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return self.weight * total
