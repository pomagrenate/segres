from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from .ema import ModelEMA


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    ema_model: Optional[ModelEMA] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    """Save training checkpoint with optimizer, scheduler, EMA, and AMP scaler state."""
    ckpt = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if ema_model is not None:
        ckpt["ema_state_dict"] = ema_model.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[ModelEMA] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint."""
    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ema_model is not None and "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return {"epoch": ckpt.get("epoch", 0), "loss": ckpt.get("loss", float("inf"))}
