# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class BaseLoss(nn.Module):
    """Base class for all loss functions."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            pred: Model predictions (logits or probabilities)
            target: Ground truth
            valid_mask: Optional valid region mask
            **kwargs: Additional arguments
            
        Returns:
            Loss value
        """
        raise NotImplementedError
    
    def _apply_valid_mask(
        self,
        loss: torch.Tensor,
        valid_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply valid mask to loss if provided."""
        if valid_mask is not None:
            loss = loss * valid_mask
            return loss.sum() / valid_mask.sum().clamp_min(1.0)
        return loss.mean()
