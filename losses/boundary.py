# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import BaseLoss


class BoundaryBCELoss(BaseLoss):
    """Boundary-aware BCE loss using Sobel edge detection."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        
        # Sobel filters
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)
    
    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel filters."""
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        edge_target = self._sobel_edges(target)
        edge_pred = self._sobel_edges(prob)
        
        loss = F.l1_loss(edge_pred, edge_target, reduction="none")
        return self.weight * self._apply_valid_mask(loss, valid_mask)


class BoundaryDiceLoss(BaseLoss):
    """Boundary-aware Dice loss."""
    
    def __init__(self, weight: float = 1.0, smooth: float = 1.0):
        super().__init__(weight)
        self.smooth = smooth
        
        # Sobel filters
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)
    
    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge map using Sobel filters."""
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        edge_target = self._sobel_edges(target)
        edge_pred = self._sobel_edges(prob)
        
        if valid_mask is not None:
            edge_pred = edge_pred * valid_mask
            edge_target = edge_target * valid_mask
        
        inter = (edge_pred * edge_target).sum(dim=(1, 2, 3))
        union = edge_pred.sum(dim=(1, 2, 3)) + edge_target.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()
        
        return self.weight * loss


class BoundaryDistLoss(BaseLoss):
    """Boundary distance loss using distance transform."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
    
    def _distance_transform(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute distance transform of binary mask."""
        # Simplified distance transform using Euclidean distance
        # In practice, use cv2.distanceTransform for better accuracy
        import cv2
        import numpy as np
        
        batch_size = mask.shape[0]
        distances = []
        
        for i in range(batch_size):
            m = mask[i, 0].cpu().numpy().astype(np.uint8)
            dist = cv2.distanceTransform(255 - m, cv2.DIST_L2, 5)
            distances.append(torch.from_numpy(dist).to(mask.device))
        
        return torch.stack(distances).unsqueeze(1)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        pred_binary = (prob > 0.5).float()
        target_binary = target
        
        dist_pred = self._distance_transform(pred_binary)
        dist_target = self._distance_transform(target_binary)
        
        loss = F.l1_loss(dist_pred, dist_target, reduction="none")
        return self.weight * self._apply_valid_mask(loss, valid_mask)
