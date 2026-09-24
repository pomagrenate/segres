# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy

from nn.modules import Conv, Bottleneck, C3k2, SPPF, Upsample, Concat


def parse_model(d: Dict, ch: List[int], verbose: bool = True) -> nn.Sequential:
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.
    
    Args:
        d: Model dictionary loaded from YAML
        ch: Input channels
        verbose: Whether to print model info
        
    Returns:
        model: PyTorch model
    """
    layers = []
    c2 = ch[-1]  # output channels
    
    # Parse backbone
    if 'backbone' in d:
        for i, (f, n, m, args) in enumerate(d['backbone']):
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                if isinstance(a, str):
                    # Don't eval string literals like 'nearest', 'bilinear'
                    try:
                        args[j] = eval(a)
                    except (NameError, SyntaxError):
                        # Keep as string if eval fails
                        args[j] = a
            
            n = n_ = max(round(n), 1) if n > 1 else n  # depth gain
            if m in (Conv, Bottleneck, C3k2, SPPF):
                c1, c2 = ch[f], args[0] if args else c2
                if c2 != c1:  # if not same
                    if c2 != c1:  # if not same
                        c2 = args[0]
                args = [c1, c2, *args[1:]]
            elif m is Upsample:
                # Upsample args: [scale_factor, mode] -> Upsample(scale_factor=..., mode=...)
                if len(args) >= 2:
                    args = [None, args[0], args[1]]  # size=None, scale_factor=args[0], mode=args[1]
                elif len(args) == 1:
                    args = [None, args[0], 'nearest']  # size=None, scale_factor=args[0], mode='nearest'
                else:
                    args = [None, 2, 'nearest']  # defaults
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
                # Concat takes dimension argument (default 1)
                if not args:
                    args = [1]  # default dimension
            else:
                c2 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)
            np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
    
    # Parse neck
    if 'neck' in d:
        for i, (f, n, m, args) in enumerate(d['neck']):
            m = eval(m) if isinstance(m, str) else m
            for j, a in enumerate(args):
                if isinstance(a, str):
                    # Don't eval string literals like 'nearest', 'bilinear'
                    try:
                        args[j] = eval(a)
                    except (NameError, SyntaxError):
                        # Keep as string if eval fails
                        args[j] = a
            
            n = n_ = max(round(n), 1) if n > 1 else n
            if m in (Conv, Bottleneck, C3k2, SPPF):
                c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
                c2 = args[0] if args else c1
                args = [c1, c2, *args[1:]]
            elif m is Upsample:
                # Upsample args: [scale_factor, mode] -> Upsample(scale_factor=..., mode=...)
                if len(args) >= 2:
                    args = [None, args[0], args[1]]  # size=None, scale_factor=args[0], mode=args[1]
                elif len(args) == 1:
                    args = [None, args[0], 'nearest']  # size=None, scale_factor=args[0], mode='nearest'
                else:
                    args = [None, 2, 'nearest']  # defaults
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
                # Concat takes dimension argument (default 1)
                if not args:
                    args = [1]  # default dimension
            else:
                c2 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            layers.append(m_)
            ch.append(c2)
    
    # Parse head
    if 'head' in d:
        for i, (f, n, m, args) in enumerate(d['head']):
            m = eval(m) if isinstance(m, str) else m
            for j, a in enumerate(args):
                if isinstance(a, str):
                    # Don't eval string literals like 'nearest', 'bilinear'
                    try:
                        args[j] = eval(a)
                    except (NameError, SyntaxError):
                        # Keep as string if eval fails
                        args[j] = a
            
            n = n_ = max(round(n), 1) if n > 1 else n
            if m in (Conv, Bottleneck, C3k2, SPPF):
                c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
                c2 = args[0] if args else c1
                args = [c1, c2, *args[1:]]
            elif m is Upsample:
                # Upsample args: [scale_factor, mode] -> Upsample(scale_factor=..., mode=...)
                if len(args) >= 2:
                    args = [None, args[0], args[1]]  # size=None, scale_factor=args[0], mode=args[1]
                elif len(args) == 1:
                    args = [None, args[0], 'nearest']  # size=None, scale_factor=args[0], mode='nearest'
                else:
                    args = [None, 2, 'nearest']  # defaults
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
                # Concat takes dimension argument (default 1)
                if not args:
                    args = [1]  # default dimension
            elif m is nn.Conv2d:
                c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
                c2 = args[0] if args else d.get('nc', 1)
                args = [c1, c2, *args[1:]]
            else:
                c2 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            layers.append(m_)
            ch.append(c2)
    
    return nn.Sequential(*layers), save


class SegmentationModel(nn.Module):
    """
    General-purpose high-resolution segmentation model built from YAML configuration.
    
    This model follows the Ultralytics architecture pattern:
    - Backbone: Feature extraction network
    - Neck: Feature fusion network (PANet/FPN)
    - Head: Segmentation head for mask prediction
    
    The model structure is defined in YAML files for flexibility.
    """
    
    def __init__(self, cfg: str | Dict | Path, ch: int = 3, nc: int = 1, verbose: bool = True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml.safe_load(open(cfg))
        self.yaml['nc'] = nc  # override nc
        self.yaml['channels'] = ch  # override channels
        
        # Build model
        self.model = parse_model(deepcopy(self.yaml), [ch], verbose=verbose)
        self.names = {i: f"{i}" for i in range(nc)}
        self.nc = nc
        
    def forward(self, x):
        """Forward pass."""
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)  # Always save output for layer connections
        return x
