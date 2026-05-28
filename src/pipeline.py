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
    可学习双线性融合
    输入：8×8 特征矩阵 F（每像素）
    输出：8×8 交互矩阵（64 维）
    """
    def __init__(self, d=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, d) * 0.1)

    def forward(self, F):
        B, n_ops, d, H, W = F.shape
        F_perm = F.permute(0, 3, 4, 1, 2)  # [B, H, W, 8, d]
        FA = torch.matmul(F_perm, self.A)
        interact = torch.matmul(FA, F_perm.transpose(-1, -2))  # [B, H, W, 8, 8]
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
            name: nn.Conv2d(1, 8, 1) for name in BAGUA_OPERATORS
        })

    def forward(self, x):
        base_maps = self.base_ops(x)
        multi_maps = []
        for name in BAGUA_OPERATORS:
            out = base_maps[name]
            # 算子输出的强度图范围差异大（1/方差可达上万），
            # 用无参数实例归一化稳定尺度，不改变相对强度关系。
            out = F.instance_norm(out)
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
