from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from .base import BaseLoss
from .region import DiceBCELoss, BCELoss, DiceLoss
from .boundary import BoundaryBCELoss, BoundaryDiceLoss, BoundaryDistLoss
from .structure import CLDiceLoss, SkeletonLoss


class SegmentationLoss(nn.Module):
    """
    Composition-based scientific segmentation loss framework.
    Combines pixel-wise region, boundary gradients, and topological connectivity losses.
    """

    def __init__(
        self,
        region: Optional[BaseLoss] = None,
        boundary: Optional[BaseLoss] = None,
        structure: Optional[BaseLoss] = None,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float = 3.0,
        cldice_weight: float = 0.2,
        cldice_warmup_epochs: int = 5,
        deep_supervision: bool = False,
        deep_supervision_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.region = region if region is not None else DiceBCELoss(
            weight=1.0,
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            bce_pos_weight=pos_weight,
        )
        self.boundary = boundary
        self.structure = structure if structure is not None else CLDiceLoss(weight=cldice_weight)

        self.cldice_warmup_epochs = cldice_warmup_epochs
        self.target_structure_weight = self.structure.weight if self.structure is not None else 0.0

        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights or [0.4, 0.3, 0.2, 0.1]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
        auxiliary: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_parts: Dict[str, float] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        # Region loss
        region_loss = self.region(pred, target, valid_mask)
        total = total + region_loss
        loss_parts["region"] = float(region_loss.detach().item())

        # Boundary loss
        if self.boundary is not None:
            bnd_loss = self.boundary(pred, target, valid_mask)
            total = total + bnd_loss
            loss_parts["boundary"] = float(bnd_loss.detach().item())

        # Topological Structure loss with warmup scheduling
        if self.structure is not None:
            if epoch < self.cldice_warmup_epochs:
                scale = 0.0
            else:
                scale = min(1.0, (epoch - self.cldice_warmup_epochs + 1) / max(1, self.cldice_warmup_epochs))
            self.structure.weight = self.target_structure_weight * scale

            if self.structure.weight > 0.0:
                struct_loss = self.structure(pred, target, valid_mask)
                total = total + struct_loss
                loss_parts["cldice"] = float(struct_loss.detach().item())
            else:
                loss_parts["cldice"] = 0.0

        # Deep supervision
        if self.deep_supervision and auxiliary is not None:
            ds_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
            for key, weight in zip(sorted(auxiliary.keys()), self.deep_supervision_weights):
                aux_pred = auxiliary[key]
                ds_loss = ds_loss + weight * self.region(aux_pred, target, valid_mask)
            total = total + ds_loss
            loss_parts["deep_supervision"] = float(ds_loss.detach().item())

        loss_parts["total"] = float(total.detach().item())
        return total, loss_parts

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SegmentationLoss":
        """Instantiate loss configuration from YAML dictionary."""
        return cls(
            bce_weight=float(config.get("bce_weight", 1.0)),
            dice_weight=float(config.get("dice_weight", 1.0)),
            pos_weight=float(config.get("pos_weight", 3.0)),
            cldice_weight=float(config.get("cldice_weight", 0.2)),
            cldice_warmup_epochs=int(config.get("cldice_warmup_epochs", 5)),
        )
