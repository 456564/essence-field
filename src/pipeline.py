"""
物理算子流水线

RGB → 8物理算子 → 投影 → 64维本质场

简化版：单投影无融合。先验证算子有效性，后续再加双投影融合。
"""

import torch
import torch.nn as nn
from .operators import PhysicalOperatorLayer

PHYSICAL_OPERATOR_NAMES = ['dong', 'jing', 'gang', 'rou', 'ju', 'san', 'yang', 'yin']


class PhysicalPipeline(nn.Module):
    """
    物理算子流水线 — 单层

    RGB [B,3,H,W]
      ↓ PhysicalOperatorLayer
    8 响应图 [B,8,H,W]
      ↓ 1×1 conv 投影 (8→1 per operator → 8-dim each)
    8×8 特征 [B,8,8,H,W]
      ↓ reshape
    64 维本质场 [B,64,H,W]
    """

    def __init__(self):
        super().__init__()
        self.operator_layer = PhysicalOperatorLayer()
        # 每个算子独立投影：1 通道 → 8 维
        self.projections = nn.ModuleDict({
            name: nn.Conv2d(1, 8, 1, bias=False)
            for name in PHYSICAL_OPERATOR_NAMES
        })
        # 投影权重初始化为正值（非负约束）
        for proj in self.projections.values():
            nn.init.uniform_(proj.weight, 0.0, 0.5)
            # 对角线初始化为 1（恒等偏好）
            with torch.no_grad():
                for i in range(min(proj.weight.shape[0], proj.weight.shape[1])):
                    proj.weight[i, i] = 1.0

    def forward(self, x):
        base = self.operator_layer(x)  # [B, 8, H, W]

        multi = []
        for i, name in enumerate(PHYSICAL_OPERATOR_NAMES):
            ch = base[:, i:i+1, :, :]                # [B, 1, H, W]
            feat = self.projections[name](ch)          # [B, 8, H, W]
            multi.append(feat)

class BaguaPipeline(nn.Module):
    """
    八卦流水线
    输入 [B, 3, H, W] → 64 维卦象场 [B, 64, H, W]
    """

    def __init__(self, d=8):
        super().__init__()
        self.operator_layer = MultiDimOperatorLayer()
        self.fusion = BilinearFusion(d=d)
        self.diffusion = None  # 由外部设置

    def forward(self, x, return_operators=False):
        """
        Args:
            x: [B, 3, H, W] 输入图像
            return_operators: 是否返回原始算子响应（用于可视化/边缘门）

        Returns:
            若 return_operators=False: field_64
            若 return_operators=True: (field_64, raw_ops_dict)
        """
        # 原始算子响应（instance_norm 前）
        raw_ops = self.operator_layer.base_ops(x)

        # 投影到多维
        multi_feat = self.operator_layer(x)    # [B, 8, 8, H, W]

        # 双线性融合 → 64 维场
        hexagram = self.fusion(multi_feat)      # [B, 64, H, W]

        # 可选：场自组织扩散
        if self.diffusion is not None:
            hexagram, _conv = self.diffusion(hexagram, raw_ops)

        if return_operators:
            return hexagram, raw_ops
        return hexagram
