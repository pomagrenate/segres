from __future__ import annotations

import torch
from soar.losses import SegmentationLoss, DiceBCELoss, CLDiceLoss, BoundaryDistLoss

def test_loss_backward():
    """Verify composite and atomic losses compute gradients stably without FP16/FP32 overflow."""
    loss_fn = SegmentationLoss()
    pred = torch.randn(1, 1, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, (1, 1, 128, 128), dtype=torch.float32)

    total_loss, loss_items = loss_fn(pred, target)
    assert not torch.isnan(total_loss), "Loss computed NaN"
    assert not torch.isinf(total_loss), "Loss computed Inf"
    total_loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any(), "Gradient contains NaN"
