from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "select_group_count",
    "autopad",
    "CBA",
    "Down",
    "LKR",
    "Ctx",
    "Fuse",
    "Agg",
    "SegHead",
    "Conv",
    "Bottleneck",
    "C3k2",
    "SPPF",
    "Upsample",
    "Concat",
)


def select_group_count(channels: int, max_groups: int = 8) -> int:
    """Determine highest divisor of channels <= max_groups to guarantee GroupNorm divisibility."""
    for g in range(min(max_groups, channels), 1, -1):
        if channels % g == 0:
            return g
    if channels > 1:
        warnings.warn(
            f"select_group_count: channel count {channels} is indivisible by any group size in 2..{max_groups}. "
            f"Falling back to num_groups=1 (LayerNorm-like behavior).",
            UserWarning,
            stacklevel=2,
        )
    return 1


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Pad to 'same' output dimensions when not specified."""
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


class CBA(nn.Module):
    """Conv2d -> GroupNorm -> Activation. Engineered for batch_size=1 stability."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = nn.GroupNorm(num_groups=select_group_count(c2), num_channels=c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Down(nn.Module):
    """Separable stride-2 downsampling: depthwise 3x3/s2 -> pointwise channel projection."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.dw = CBA(c1, c1, 3, 2, g=c1, act=False)
        self.pw = CBA(c1, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class LKR(nn.Module):
    """
    Large-Kernel Residual block (SOAR backbone & decoder unit).
    Depthwise kxk -> 1x1 expand -> SiLU -> 1x1 project, with identity residual connection.
    Norm projection weight is zero-initialized so deep stacks start as an exact identity map.
    """

    def __init__(self, c: int, k: int = 7, e: float = 2.0):
        super().__init__()
        h = max(int(c * e), 8)
        self.dw = CBA(c, c, k, 1, g=c, act=False)
        self.pw1 = CBA(c, h, 1)
        self.pw2 = CBA(h, c, 1, act=False)
        nn.init.zeros_(self.pw2.norm.weight)
        if self.pw2.norm.bias is not None:
            nn.init.zeros_(self.pw2.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.pw1(self.dw(x)))


class Ctx(nn.Module):
    """
    Deep-stage multi-scale context module.
    Parallel dilated depthwise convolutions with a global average pooling channel-attention gate.
    """

    def __init__(self, c1: int, c2: int, dilations: Sequence[int] = (1, 3, 5)):
        super().__init__()
        h = c2 // 2
        self.cv1 = CBA(c1, h, 1)
        self.branches = nn.ModuleList(CBA(h, h, 3, 1, g=h, d=int(d), act=False) for d in dilations)
        self.cv2 = CBA(h * (1 + len(dilations)), c2, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c1, c2, 1), nn.Sigmoid())
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = self.cv2(torch.cat([y] + [b(y) for b in self.branches], 1))
        y = y * self.gate(x)
        return x + y if self.add else y


class Fuse(nn.Module):
    """
    Gated top-down cross-attention fusion of coarse semantic context and fine spatial detail.
    Input: [low (coarse), skip (fine)].
    Learns spatial and channel-wise convex combination weights between detail and context.
    """

    def __init__(self, chs: Sequence[int], c2: int):
        super().__init__()
        c_low, c_skip = chs
        self.low = CBA(c_low, c2, 1, act=False)
        self.skip = CBA(c_skip, c2, 1, act=False)
        self.gate = nn.Sequential(CBA(c2, c2, 3, 1, g=c2, act=False), nn.Conv2d(c2, c2, 1), nn.Sigmoid())
        self.out = nn.Sequential(CBA(c2, c2, 3, 1, g=c2, act=False), CBA(c2, c2, 1))

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        low, skip = xs
        a = self.skip(skip)
        b = F.interpolate(self.low(low), size=a.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(a + b)
        return self.out(g * a + (1.0 - g) * b)


class Agg(nn.Module):
    """
    Multi-scale semantic aggregation at stride 4.
    Input: list of feature maps with highest resolution first.
    Projects each level at native resolution, upsamples to target size, concatenates, and fuses.
    """

    def __init__(self, chs: Sequence[int], c2: int):
        super().__init__()
        p = max(c2 // 2, 8)
        self.proj = nn.ModuleList(CBA(c, p, 1) for c in chs)
        self.fuse = CBA(p * len(chs), c2, 1)

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        size = xs[0].shape[-2:]
        ys = []
        for m, x in zip(self.proj, xs):
            y = m(x)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            ys.append(y)
        return self.fuse(torch.cat(ys, 1))


class SegHead(nn.Module):
    """
    Sub-pixel segmentation head.
    Predicts nc*r*r channels at stride r and rearranges them to full resolution (stride 1)
    with PixelShuffle, eliminating blurry bilinear logit upsampling.
    Initializes bias using prior probability to prevent background gradient avalanche.
    """

    def __init__(self, c1: int, nc: int, mid: int = 32, r: int = 2, prior: float | None = 0.01):
        super().__init__()
        self.refine = nn.Sequential(CBA(c1, c1, 3, 1, g=c1, act=False), CBA(c1, mid, 1))
        self.pred = nn.Conv2d(mid, nc * r * r, 1)
        self.shuffle = nn.PixelShuffle(r)
        if prior is not None and prior > 0.0:
            nn.init.constant_(self.pred.bias, -math.log((1.0 - prior) / prior))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.pred(self.refine(x)))


# Backwards compatibility wrappers
class Conv(CBA):
    """Alias for CBA."""
    pass


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: int = 3, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = CBA(c1, c_, k, 1)
        self.cv2 = CBA(c_, c2, k, 1, g=1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """CSP bottleneck block."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = CBA(c1, self.c_, 1, 1)
        self.cv2 = CBA(c1, self.c_, 1, 1)
        self.cv3 = CBA(2 * self.c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = CBA(c1, c_, 1, 1)
        self.cv2 = CBA(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Upsample(nn.Module):
    """Upsample layer."""

    def __init__(self, size: Optional[Union[int, Tuple[int, int]]] = None, scale_factor: float = 2.0, mode: str = "nearest"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=None if self.size else self.scale_factor,
            mode=self.mode,
            align_corners=False if self.mode in ("bilinear", "bicubic") else None,
        )


class Concat(nn.Module):
    """Concatenate along channel dimension."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)
