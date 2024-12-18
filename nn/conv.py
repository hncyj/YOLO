import math
import numpy as np
import torch
import torch.nn as nn

__all__ = ()

def autopad(k, p=None, d=1):
    """
    Pad input to make output the same shape.\\
    d = 1 means no dialtion\\
    k = k + (k - 1) * (d - 1) = (k - 1) * d + 1.
    
    Args:
        k (int): kernal size
        p (bool, optional): padding
        d (int, optional): dilation
    """
    if d > 1:
        k = (k - 1) * d + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    
    return p


class Conv(nn.modules):
    default_act = nn.SiLU()
    def __init__(self, ch_in, ch_out, k=1, s=1, p=None, g=1, d=1, act=True):
        super.__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, autopad(k, p , d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
    
    
    

