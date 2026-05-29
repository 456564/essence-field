"""
八卦→64卦 流水线

核心数据结构：64 维卦象场 [B, 64, H, W]
  - 每个像素有自己的 64 维向量
  - 同一物质的像素有相似的 64 维向量
  - 64 维向量不是图片级别的描述，是像素级别的

任何取全图均值的做法都是错误的。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .operators import BAGUA_OPERATORS

# 算子层选择：colormod 版用 nn.Module 实现
try:
    from .operators import ColorModulatedOperatorLayer as _BaseOps
    _HAS_COLORMOD = True
except ImportError:
    from .operators import BaguaOperatorLayer as _BaseOps
    _HAS_COLORMOD = False


class BilinearFusion(nn.Module):
    """
    双投影双线性融合 — 上卦/下卦各独立投影，生成64卦有序对。
    field[i*8+j] = F_up[i]·F_dn[j], 非对称, 卦纯可追溯。
    """
    def __init__(self, d=8):
        super().__init__()
        self.W_up = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.02)
        self.W_dn = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.02)

    def forward(self, F):
        B, n_ops, d, H, W = F.shape
        Fu = torch.einsum('bnihw,km->bnmhw', F, self.W_up)  # [B,8,8,H,W] 上卦投影
        Fd = torch.einsum('bnihw,km->bnmhw', F, self.W_dn)  # [B,8,8,H,W] 下卦投影
        Fu = Fu.permute(0, 3, 4, 1, 2)  # [B,H,W,8,8]
        Fd = Fd.permute(0, 3, 4, 1, 2)
        interact = torch.matmul(Fu, Fd.transpose(-1, -2))  # [B,H,W,8,8]
        hexagram = interact.reshape(B, H, W, n_ops * n_ops).permute(0, 3, 1, 2)
        return hexagram


class MultiDimOperatorLayer(nn.Module):
    """
    多维算子层
    每个算子输出 8 维特征 → [B, 8, 8, H, W]
    """
    def __init__(self):
        super().__init__()
        self.base_ops = _BaseOps()
        self.projections = nn.ModuleDict({
            name: nn.Conv2d(1, 8, 1, bias=False) for name in BAGUA_OPERATORS
        })

    def forward(self, x):
        base_maps = self.base_ops(x)
        multi_maps = []
        for name in BAGUA_OPERATORS:
            out = base_maps[name]
            # /amax: 算子内部归一化到 [0,1]
            out = out / (out.amax(dim=(2, 3), keepdim=True) + 1e-6)
            # /q90均衡: 活跃像素代表值≈1，稀疏密集均公平
            out = out / (out.quantile(0.9) + 1e-6)
            feat = self.projections[name](out)
            multi_maps.append(feat)
        return torch.stack(multi_maps, dim=1)


class BaguaPipeline(nn.Module):
    """
    八卦流水线
    输入 [B, 3, H, W] → 64 维卦象场 [B, 64, H, W]
    """
    def __init__(self, d=8):
        super().__init__()
        self.operator_layer = MultiDimOperatorLayer()
        self.fusion = BilinearFusion(d=d)

    def forward(self, x):
        multi_feat = self.operator_layer(x)    # [B, 8, 8, H, W]
        hexagram = self.fusion(multi_feat)      # [B, 64, H, W]
        return hexagram
