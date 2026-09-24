from __future__ import annotations

import copy
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average (EMA) of model weights."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.device = device
        self.shadow_model = copy.deepcopy(model).eval()
        if device is not None:
            self.shadow_model.to(device)

        for p in self.shadow_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow model with EMA."""
        d = self.decay
        for s_param, m_param in zip(self.shadow_model.parameters(), model.parameters()):
            s_param.data.mul_(d).add_(m_param.data.to(s_param.device), alpha=(1.0 - d))

        for s_buf, m_buf in zip(self.shadow_model.buffers(), model.buffers()):
            s_buf.copy_(m_buf.to(s_buf.device))

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "decay": self.decay,
            "shadow_state_dict": self.shadow_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow_model.load_state_dict(state_dict["shadow_state_dict"])
