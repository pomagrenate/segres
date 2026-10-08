from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


@dataclass(frozen=True)
class DeviceSelection:
    """Requested and resolved compute device for a training run."""

    requested: str
    device: torch.device
    fell_back: bool = False


def resolve_device(requested_device: Union[str, torch.device]) -> DeviceSelection:
    """Resolve CPU/CUDA requests without silently masking invalid device names."""
    requested = str(requested_device).strip().lower()
    if not requested:
        raise ValueError("Device must be 'cpu', 'cuda', or a CUDA index such as 'cuda:0'.")

    try:
        candidate = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"Unsupported device {requested_device!r}; expected 'cpu', 'cuda', or 'cuda:N'."
        ) from exc

    if candidate.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device {requested_device!r}; expected 'cpu', 'cuda', or 'cuda:N'."
        )

    if candidate.type == "cuda":
        if not torch.cuda.is_available():
            return DeviceSelection(requested=requested, device=torch.device("cpu"), fell_back=True)
        device_count = torch.cuda.device_count()
        if candidate.index is not None and candidate.index >= device_count:
            raise ValueError(
                f"CUDA device index {candidate.index} is unavailable; "
                f"detected {device_count} CUDA device(s)."
            )

    return DeviceSelection(requested=requested, device=candidate, fell_back=False)


def format_device_selection(selection: DeviceSelection) -> str:
    """Build a concise user-facing description of the actual training device."""
    if selection.device.type == "cuda":
        index = selection.device.index
        if index is None:
            index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(index)
        cuda_runtime = torch.version.cuda or "unknown"
        return (
            f"[Device] Using CUDA GPU: {gpu_name} "
            f"({selection.device}, PyTorch {torch.__version__}, CUDA runtime {cuda_runtime})."
        )

    if selection.fell_back:
        cuda_runtime = torch.version.cuda or "none (CPU-only PyTorch build)"
        return (
            "[Device] WARNING: CUDA was requested but is unavailable; falling back to CPU. "
            f"PyTorch {torch.__version__}; CUDA runtime: {cuda_runtime}."
        )

    return "[Device] Using CPU (explicitly requested)."


def announce_device(selection: DeviceSelection) -> None:
    """Print the resolved training device once at startup."""
    print(format_device_selection(selection), flush=True)
