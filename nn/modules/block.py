# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn


def autopad(k):
    """Pad to 'same' output dimensions when not specified."""
    if isinstance(k, int):
        return k // 2
    if isinstance(k, (list, tuple)):
        return [x // 2 for x in k]
    return int(k) // 2


class Conv(nn.Module):
    """Standard convolution with GroupNorm and SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        c1, c2, k, s, g = int(c1), int(c2), int(k), int(s), int(g)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False)
        self.norm = nn.GroupNorm(min(8, c2), c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k, 1)
        self.cv2 = Conv(c_, c2, k, 1, g=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """C3k2 module with CSP bottleneck - simplified version."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.cv2 = Conv(c1, self.c_, 1, 1)
        self.cv3 = Conv(2 * self.c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Upsample(nn.Module):
    """Upsample layer."""
    def __init__(self, size=None, scale_factor=2, mode='nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
