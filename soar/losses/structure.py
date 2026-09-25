from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseLoss


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Soft morphological erosion via directional min-pooling."""
    neg_x = -x
    p1 = -F.max_pool2d(neg_x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(neg_x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Soft morphological dilation via max-pooling."""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_skeletonize(x: torch.Tensor, n_iter: int = 3) -> torch.Tensor:
    """Differentiable soft skeletonization for topological connectivity preservation."""
    skel = torch.zeros_like(x)
    curr = x
    for _ in range(n_iter):
        eroded = soft_erode(curr)
        # open(curr) = dilate(erode(curr)) = dilate(eroded)
        opened = soft_dilate(eroded)
        delta = F.relu(curr - opened)
        skel = skel + delta * (1.0 - skel)
        curr = eroded
    return skel


class CLDiceLoss(BaseLoss):
    """
    Continuous clDice loss for curvilinear and thin topological structure segmentation.
    Enforces skeleton-level precision and sensitivity with strict float32 numerical guarantees.
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
        prob = torch.sigmoid(pred).float()
        target = target.float()

        if valid_mask is not None:
            vmask = valid_mask.float()
            prob = prob * vmask
            target = target * vmask

        skel_pred = soft_skeletonize(prob, self.n_iter)
        with torch.no_grad():
            skel_true = soft_skeletonize(target, self.n_iter)

        dims = tuple(range(1, prob.ndim))
        t_prec = (torch.sum(skel_pred * target, dim=dims) + self.smooth) / (
            torch.sum(skel_pred, dim=dims) + self.smooth
        ).clamp_min(1e-7)
        t_sens = (torch.sum(skel_true * prob, dim=dims) + self.smooth) / (
            torch.sum(skel_true, dim=dims) + self.smooth
        ).clamp_min(1e-7)

        cldice = (2.0 * t_prec * t_sens) / (t_prec + t_sens).clamp_min(1e-7)
        return self.weight * (1.0 - cldice.mean())


class SkeletonLoss(BaseLoss):
    """Direct L1 penalty on soft topological skeletons."""

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
        prob = torch.sigmoid(pred).float()
        target = target.float()

        if valid_mask is not None:
            vmask = valid_mask.float()
            prob = prob * vmask
            target = target * vmask

        skel_pred = soft_skeletonize(prob, self.n_iter)
        with torch.no_grad():
            skel_true = soft_skeletonize(target, self.n_iter)

        loss = F.l1_loss(skel_pred, skel_true, reduction="none")
        return self.weight * self._apply_valid_mask(loss, valid_mask)
