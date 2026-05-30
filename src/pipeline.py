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

        # [B, 8, 8, H, W] → [B, 64, H, W]
        field = torch.cat([m.unsqueeze(1) for m in multi], dim=1)
        B, n_ops, d, H, W = field.shape
        field = field.reshape(B, n_ops * d, H, W)
        return field

    def clamp_weights(self):
        """投影权重 clamp ≥ 0，保持全管线非负"""
        for proj in self.projections.values():
            proj.weight.data.clamp_(min=0.0)
