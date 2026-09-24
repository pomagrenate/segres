from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseLoss


class BCELoss(BaseLoss):
    """Numerically stable Binary Cross-Entropy loss with optional positive weighting."""

    def __init__(self, weight: float = 1.0, pos_weight: float = 1.0):
        super().__init__(weight)
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        pw = self.pos_weight.to(device=pred.device, dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(
            pred.float(),
            target.float(),
            pos_weight=pw,
            reduction="none",
        )
        return self.weight * self._apply_valid_mask(loss, valid_mask)


class DiceLoss(BaseLoss):
    """Soft Dice loss with float32 spatial reduction to prevent FP16 overflow at native resolution."""

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
        prob = torch.sigmoid(pred).float()
        target = target.float()

        if valid_mask is not None:
            vmask = valid_mask.float()
            prob = prob * vmask
            target = target * vmask

        dims = tuple(range(1, prob.ndim))
        intersection = torch.sum(prob * target, dim=dims)
        cardinality = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(1e-7)
        return self.weight * (1.0 - dice.mean())


class FocalLoss(BaseLoss):
    """Focal loss for extreme foreground-background class imbalance."""

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
        prob = torch.sigmoid(pred).float()
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(pred.float(), target, reduction="none")
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        focal_weight = (1.0 - pt).clamp_min(0.0) ** self.gamma

        loss = self.alpha * focal_weight * bce
        return self.weight * self._apply_valid_mask(loss, valid_mask)


class TverskyLoss(BaseLoss):
    """Tversky loss controlling false-positive and false-negative penalties."""

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
        prob = torch.sigmoid(pred).float()
        target = target.float()

        if valid_mask is not None:
            vmask = valid_mask.float()
            prob = prob * vmask
            target = target * vmask

        dims = tuple(range(1, prob.ndim))
        tp = torch.sum(prob * target, dim=dims)
        fp = torch.sum(prob * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - prob) * target, dim=dims)

        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denominator.clamp_min(1e-7)
        return self.weight * (1.0 - tversky.mean())


class FocalTverskyLoss(BaseLoss):
    """Focal Tversky loss focusing on challenging topological structures."""

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ):
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
        prob = torch.sigmoid(pred).float()
        target = target.float()

        if valid_mask is not None:
            vmask = valid_mask.float()
            prob = prob * vmask
            target = target * vmask

        dims = tuple(range(1, prob.ndim))
        tp = torch.sum(prob * target, dim=dims)
        fp = torch.sum(prob * (1.0 - target), dim=dims)
        fn = torch.sum((1.0 - prob) * target, dim=dims)

        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denominator.clamp_min(1e-7)
        focal_tversky = (1.0 - tversky).clamp_min(0.0) ** self.gamma
        return self.weight * focal_tversky.mean()


class DiceBCELoss(BaseLoss):
    """Composite Dice and Binary Cross-Entropy baseline loss."""

    def __init__(
        self,
        weight: float = 1.0,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        bce_pos_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__(weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss_fn = BCELoss(weight=1.0, pos_weight=bce_pos_weight)
        self.dice_loss_fn = DiceLoss(weight=1.0, smooth=smooth)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        bce = self.bce_loss_fn(pred, target, valid_mask)
        dice = self.dice_loss_fn(pred, target, valid_mask)
        return self.weight * (self.bce_weight * bce + self.dice_weight * dice)
