# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from .base import BaseLoss
from .region import DiceBCELoss, BCELoss, DiceLoss, FocalLoss, TverskyLoss, FocalTverskyLoss
from .boundary import BoundaryBCELoss, BoundaryDiceLoss, BoundaryDistLoss
from .structure import CLDiceLoss, SkeletonLoss


class SegmentationLoss(nn.Module):
    """
    Composition-based segmentation loss.
    
    Allows flexible combination of region, boundary, and structure losses.
    Default: BCE + Dice (strong generic baseline).
    """
    
    def __init__(
        self,
        region: Optional[BaseLoss] = None,
        boundary: Optional[BaseLoss] = None,
        structure: Optional[BaseLoss] = None,
        deep_supervision: bool = False,
        deep_supervision_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        # Default: Dice + BCE
        self.region = region if region is not None else DiceBCELoss(weight=1.0)
        self.boundary = boundary
        self.structure = structure
        
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights or [0.4, 0.3, 0.2, 0.1]
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
        auxiliary: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute composite loss.
        
        Args:
            pred: Model predictions (logits)
            target: Ground truth masks
            valid_mask: Valid region mask
            epoch: Current training epoch (for loss scheduling)
            auxiliary: Optional auxiliary outputs for deep supervision
            
        Returns:
            total_loss: Combined loss
            loss_parts: Dictionary of individual loss components
        """
        loss_parts = {}
        total = 0.0
        
        # Region loss (always active)
        region_loss = self.region(pred, target, valid_mask)
        total += region_loss
        loss_parts["region"] = region_loss.detach()
        
        # Boundary loss (optional)
        if self.boundary is not None:
            boundary_loss = self.boundary(pred, target, valid_mask)
            total += boundary_loss
            loss_parts["boundary"] = boundary_loss.detach()
        
        # Structure loss (optional)
        if self.structure is not None:
            structure_loss = self.structure(pred, target, valid_mask)
            total += structure_loss
            loss_parts["structure"] = structure_loss.detach()
        
        # Deep supervision (optional)
        if self.deep_supervision and auxiliary is not None:
            ds_loss = 0.0
            for i, (key, weight) in enumerate(zip(sorted(auxiliary.keys()), self.deep_supervision_weights)):
                aux_pred = auxiliary[key]
                aux_loss = self.region(aux_pred, target, valid_mask)
                ds_loss += weight * aux_loss
            
            total += ds_loss
            loss_parts["deep_supervision"] = ds_loss.detach()
        
        loss_parts["total"] = total.detach()
        return total, loss_parts
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SegmentationLoss":
        """
        Create loss from configuration dictionary.
        
        Example config:
        {
            "region": {"type": "dice_bce", "weight": 1.0},
            "boundary": {"type": "boundary_bce", "weight": 0.2},
            "structure": {"type": "cldice", "weight": 0.3},
            "deep_supervision": true
        }
        """
        region = None
        boundary = None
        structure = None
        
        # Parse region loss
        if "region" in config:
            region_cfg = config["region"]
            region_type = region_cfg.get("type", "dice_bce")
            region_weight = region_cfg.get("weight", 1.0)
            
            if region_type == "bce":
                region = BCELoss(weight=region_weight, pos_weight=region_cfg.get("pos_weight", 1.0))
            elif region_type == "dice":
                region = DiceLoss(weight=region_weight, smooth=region_cfg.get("smooth", 1.0))
            elif region_type == "focal":
                region = FocalLoss(weight=region_weight, gamma=region_cfg.get("gamma", 2.0), alpha=region_cfg.get("alpha", 0.25))
            elif region_type == "tversky":
                region = TverskyLoss(weight=region_weight, alpha=region_cfg.get("alpha", 0.5), beta=region_cfg.get("beta", 0.5))
            elif region_type == "focal_tversky":
                region = FocalTverskyLoss(weight=region_weight, alpha=region_cfg.get("alpha", 0.5), beta=region_cfg.get("beta", 0.5), gamma=region_cfg.get("gamma", 2.0))
            elif region_type == "dice_bce":
                region = DiceBCELoss(weight=region_weight, dice_weight=region_cfg.get("dice_weight", 1.0), bce_weight=region_cfg.get("bce_weight", 1.0))
        
        # Parse boundary loss
        if "boundary" in config and config["boundary"].get("enabled", False):
            boundary_cfg = config["boundary"]
            boundary_type = boundary_cfg.get("type", "boundary_bce")
            boundary_weight = boundary_cfg.get("weight", 0.2)
            
            if boundary_type == "boundary_bce":
                boundary = BoundaryBCELoss(weight=boundary_weight)
            elif boundary_type == "boundary_dice":
                boundary = BoundaryDiceLoss(weight=boundary_weight, smooth=boundary_cfg.get("smooth", 1.0))
            elif boundary_type == "boundary_dist":
                boundary = BoundaryDistLoss(weight=boundary_weight)
        
        # Parse structure loss
        if "structure" in config and config["structure"].get("enabled", False):
            structure_cfg = config["structure"]
            structure_type = structure_cfg.get("type", "cldice")
            structure_weight = structure_cfg.get("weight", 0.3)
            
            if structure_type == "cldice":
                structure = CLDiceLoss(weight=structure_weight, n_iter=structure_cfg.get("n_iter", 3))
            elif structure_type == "skeleton":
                structure = SkeletonLoss(weight=structure_weight, n_iter=structure_cfg.get("n_iter", 3))
        
        # Deep supervision
        deep_supervision = config.get("deep_supervision", False)
        deep_supervision_weights = config.get("deep_supervision_weights", None)
        
        return cls(
            region=region,
            boundary=boundary,
            structure=structure,
            deep_supervision=deep_supervision,
            deep_supervision_weights=deep_supervision_weights,
        )
