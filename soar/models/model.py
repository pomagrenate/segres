from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..nn.modules import (
    CBA,
    Down,
    LKR,
    Ctx,
    Fuse,
    Agg,
    SegHead,
    Conv,
    Bottleneck,
    C3k2,
    SPPF,
    Upsample,
    Concat,
)

# Registry mapping: name -> (class, kind)
#   ch    : cls(c1, c2, *args)        c2 = args[0] * width
#   keep  : cls(c1, *args)            channel-preserving, repeated n * depth times
#   multi : cls([c1, ...], c2, *args) multi-input, c2 = args[0] * width
#   head  : cls(c1, nc, mid, *args)   mid = args[0] * width
REGISTRY: Dict[str, Tuple[Any, str]] = {
    "CBA": (CBA, "ch"),
    "Down": (Down, "ch"),
    "Ctx": (Ctx, "ch"),
    "LKR": (LKR, "keep"),
    "Fuse": (Fuse, "multi"),
    "Agg": (Agg, "multi"),
    "SegHead": (SegHead, "head"),
    "Conv": (Conv, "ch"),
    "Bottleneck": (Bottleneck, "ch"),
    "C3k2": (C3k2, "ch"),
    "SPPF": (SPPF, "ch"),
    "Upsample": (Upsample, "ch"),
    "Concat": (Concat, "multi"),
    "nn.Conv2d": (nn.Conv2d, "ch"),
}

SECTIONS = ("backbone", "decoder", "neck", "head")
DIVISOR = 32  # Maximum network downsampling factor


def make_divisible(x: float, divisor: int = 8) -> int:
    """Ensure channel dimension is cleanly divisible by divisor."""
    return max(divisor, int(-(-x // divisor) * divisor))


def guess_scale(cfg: Union[str, Path]) -> Optional[str]:
    """Infer scale variant ('n', 's', 'm', 'l', 'x') from filename."""
    if isinstance(cfg, (str, Path)):
        stem = Path(cfg).stem.lower()
        m = re.search(r"soar\d*([nsmlx])$", stem) or re.search(r"soar_([nsmlx])", stem) or re.search(r"soar_(nano|small|medium|large|xlarge)", stem)
        if m:
            val = m.group(1)
            mapping = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
            return mapping.get(val, val)
    return None


def parse_model(d: Dict[str, Any], ch: int = 3) -> Tuple[nn.Sequential, List[int], List[int]]:
    """
    Parse declarative model dictionary into an instantiated nn.Sequential layer hierarchy.
    Returns (layers, indices of intermediate outputs to retain, output channels per layer).
    """
    nc = d.get("nc", 1)
    depth = d.get("depth_multiple", 1.0)
    width = d.get("width_multiple", 1.0)
    max_ch = d.get("max_channels", 1024)

    def wc(c: int) -> int:
        return make_divisible(min(c, max_ch) * width)

    layers: List[nn.Module] = []
    out_ch: List[int] = []
    save: Set[int] = set()
    c_prev = ch

    for section in SECTIONS:
        if section not in d:
            continue
        for f, n, name, args in d[section]:
            i = len(layers)
            if name not in REGISTRY:
                raise KeyError(f"Layer {i}: unknown module '{name}'. Supported: {sorted(REGISTRY.keys())}")
            cls, kind = REGISTRY[name]

            multi_in = isinstance(f, (list, tuple))
            if multi_in != (kind == "multi"):
                raise ValueError(f"Layer {i} ({name}): {'requires' if kind == 'multi' else 'does not accept'} multi-input list.")

            refs = list(f) if multi_in else [f]
            for j in refs:
                if not (j == -1 or 0 <= j < i):
                    raise ValueError(f"Layer {i} ({name}): invalid 'from' index {j}")

            def cin(j: int) -> int:
                return c_prev if j == -1 else out_ch[j]

            args = list(args) if args else []
            if kind == "keep":
                n = max(round(n * depth), 1) if n > 1 else n
            elif n != 1 and name not in ("C3k2", "Bottleneck"):
                raise ValueError(f"Layer {i} ({name}): repeats must be 1 for non-repetitive blocks.")

            if kind == "ch":
                if name == "Upsample":
                    c1, c2 = cin(f), cin(f)
                    if len(args) >= 2:
                        a = [None, args[0], args[1]]
                    elif len(args) == 1:
                        a = [None, args[0], "nearest"]
                    else:
                        a = [None, 2, "nearest"]
                elif name == "nn.Conv2d":
                    c1 = cin(f)
                    c2 = args[0] if args else nc
                    a = [c1, c2, *args[1:]]
                else:
                    c1 = cin(f)
                    c2 = wc(args[0]) if args else c1
                    a = [c1, c2, *args[1:]]
            elif kind == "keep":
                c1 = c2 = cin(f)
                a = [c1, *args]
            elif kind == "multi":
                c1_list = [cin(j) for j in refs]
                c2 = wc(args[0]) if args else sum(c1_list)
                a = [c1_list, c2, *args[1:]]
            else:  # head
                c1 = cin(f)
                c2 = nc
                mid_channels = wc(args[0]) if args else 32
                a = [c1, nc, mid_channels, *args[1:]]

            m = nn.Sequential(*(cls(*a) for _ in range(n))) if n > 1 else cls(*a)
            m.i, m.f, m.mname, m.n = i, f, name, n
            m.np = sum(p.numel() for p in m.parameters())
            layers.append(m)
            out_ch.append(c2)
            c_prev = c2
            save.update(j for j in refs if j != -1)

    return nn.Sequential(*layers), sorted(save), out_ch


class SegmentationModel(nn.Module):
    """
    SOAR resolution-preserving segmentation network.
    Constructs multi-scale large-kernel backbones, top-down gated decoders, and sub-pixel PixelShuffle heads.
    """

    def __init__(
        self,
        cfg: Union[str, Dict[str, Any], Path] = "configs/models/soar_medium1.yaml",
        ch: int = 3,
        nc: Optional[int] = None,
        scale: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = deepcopy(cfg)
        else:
            cfg_path = Path(cfg)
            if not cfg_path.exists():
                alt = Path(str(cfg).replace("cfg/", "configs/")) if "cfg/" in str(cfg) else Path(str(cfg).replace("configs/", "cfg/"))
                if alt.exists():
                    cfg_path = alt
            with open(cfg_path, "r", encoding="utf-8") as fh:
                self.yaml = yaml.safe_load(fh)

        if nc is not None:
            self.yaml["nc"] = nc
        self.yaml["channels"] = ch

        scales = self.yaml.get("scales")
        if scales:
            inferred_scale = scale or guess_scale(cfg) or self.yaml.get("scale", "m")
            if inferred_scale not in scales:
                inferred_scale = "m"
            depth, width = scales[inferred_scale][:2]
            self.yaml.update(depth_multiple=depth, width_multiple=width)
            self.scale = inferred_scale
        else:
            self.scale = scale or guess_scale(cfg) or self.yaml.get("scale", "custom")

        self.nc = self.yaml.get("nc", 1)
        self.names = {i: f"{i}" for i in range(self.nc)}
        self.model, self.save, self.out_ch = parse_model(self.yaml, ch)

        # Precompute reference counts for eager activation cleanup
        counts: Dict[int, int] = {}
        for m in self.model:
            if m.f != -1:
                refs = list(m.f) if isinstance(m.f, (list, tuple)) else [m.f]
                for r in refs:
                    if r != -1:
                        counts[r] = counts.get(r, 0) + 1
        self._consumer_counts = counts

        self._check(ch, verbose)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with auto-padding to multiples of 32 and eager activation deallocation.
        Returns full-resolution logits matching input spatial dimensions (B, nc, H, W).
        """
        h, w = x.shape[-2:]
        ph = (-h) % DIVISOR
        pw = (-w) % DIVISOR
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if (ph < h and pw < w) else "replicate")

        rem = dict(self._consumer_counts)
        y: List[Optional[torch.Tensor]] = [None] * len(self.model)

        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = y[m.f]
                    rem[m.f] -= 1
                    if rem[m.f] == 0:
                        y[m.f] = None
                else:
                    x_in = []
                    for j in m.f:
                        if j == -1:
                            x_in.append(x)
                        else:
                            x_in.append(y[j])
                            rem[j] -= 1
                            if rem[j] == 0:
                                y[j] = None
                x = x_in

            x = m(x)
            if m.i in self.save:
                y[m.i] = x

        return x[..., :h, :w]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def _check(self, ch: int, verbose: bool) -> None:
        """Verify output stride integrity at build time and display architecture metrics."""
        size = 64
        shapes: Dict[int, Tuple[int, ...]] = {}
        hooks = [
            m.register_forward_hook(lambda _m, _i, o, k=m.i: shapes.__setitem__(k, tuple(o.shape[1:])))
            for m in self.model
        ]
        was_training = self.training
        self.eval()
        try:
            out = self.forward(torch.zeros(1, ch, size, size))
        finally:
            for hk in hooks:
                hk.remove()
            self.train(was_training)

        if tuple(out.shape) != (1, self.nc, size, size):
            raise RuntimeError(
                f"SOAR output shape is {tuple(out.shape)}, expected {(1, self.nc, size, size)}. "
                f"Check SegHead 'r' stride configuration."
            )

        if verbose:
            print(f"{'idx':>3} {'from':>12} {'n':>2} {'module':<10} {'params':>10}  {'out (C,stride)':<14}")
            for m in self.model:
                c, hh = shapes[m.i][0], shapes[m.i][1]
                print(f"{m.i:>3} {str(m.f):>12} {m.n:>2} {m.mname:<10} {m.np:>10,}  ({c}, s{size // hh})")
            print(f"SOAR1-{self.scale}: {len(self.model)} layers, {self.n_params() / 1e6:.2f}M parameters\n")
