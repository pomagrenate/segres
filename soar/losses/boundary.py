from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseLoss


class BoundaryBCELoss(BaseLoss):
    """Boundary-aware loss using Sobel edge detection."""

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(device=x.device, dtype=torch.float32)
        sy = self.sobel_y.to(device=x.device, dtype=torch.float32)
        x_f32 = x.float()
        gx = F.conv2d(x_f32, sx, padding=1)
        gy = F.conv2d(x_f32, sy, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred).float()
        with torch.no_grad():
            edge_target = self._sobel_edges(target)
        edge_pred = self._sobel_edges(prob)

        loss = F.l1_loss(edge_pred, edge_target, reduction="none")
        return self.weight * self._apply_valid_mask(loss, valid_mask)


class BoundaryDiceLoss(BaseLoss):
    """Boundary Dice loss on spatial gradient maps."""

    def __init__(self, weight: float = 1.0, smooth: float = 1.0):
        super().__init__(weight)
        self.smooth = smooth
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(device=x.device, dtype=torch.float32)
        sy = self.sobel_y.to(device=x.device, dtype=torch.float32)
        x_f32 = x.float()
        gx = F.conv2d(x_f32, sx, padding=1)
        gy = F.conv2d(x_f32, sy, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred).float()
        with torch.no_grad():
            edge_target = self._sobel_edges(target)
        edge_pred = self._sobel_edges(prob)

        if valid_mask is not None:
            vmask = valid_mask.float()
            edge_pred = edge_pred * vmask
            edge_target = edge_target * vmask

        dims = tuple(range(1, edge_pred.ndim))
        inter = torch.sum(edge_pred * edge_target, dim=dims)
        cardinality = torch.sum(edge_pred, dim=dims) + torch.sum(edge_target, dim=dims)
        dice = (2.0 * inter + self.smooth) / (cardinality + self.smooth).clamp_min(1e-7)
        return self.weight * (1.0 - dice.mean())


class BoundaryDistLoss(BaseLoss):
    """
    Differentiable Signed Distance Transform Boundary Loss (Kervadec et al.).
    Computes distance transforms on target ground truth under no_grad and integrates with
    predicted probability fields to preserve strict differentiability.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    @staticmethod
    def _compute_sdf(target_np: np.ndarray) -> np.ndarray:
        """Compute signed distance field where foreground boundary is zero."""
        pos = target_np > 0.5
        neg = ~pos
        if not np.any(pos):
            return np.ones_like(target_np, dtype=np.float32)
        if not np.any(neg):
            return -np.ones_like(target_np, dtype=np.float32)
        d_out = distance_transform_edt(neg)
        d_in = distance_transform_edt(pos)
        return (d_out - d_in).astype(np.float32)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred).float()
        batch_size = target.shape[0]

        with torch.no_grad():
            target_cpu = target.detach().cpu().numpy()
            sdfs = [self._compute_sdf(target_cpu[b, 0]) for b in range(batch_size)]
            sdf_tensor = torch.from_numpy(np.stack(sdfs, axis=0)).unsqueeze(1).to(device=pred.device, dtype=torch.float32)

        boundary_penalty = prob * sdf_tensor

        if valid_mask is not None:
            vmask = valid_mask.float()
            boundary_penalty = boundary_penalty * vmask
            return self.weight * (boundary_penalty.sum() / vmask.sum().clamp_min(1.0))

        return self.weight * boundary_penalty.mean()
