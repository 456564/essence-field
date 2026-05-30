"""
物理算子流水线 v2 — 4算子 → 16维本质空间

RGB → 4物理算子 → 投影 → 4×4=16维本质场
"""

import torch
import torch.nn as nn
from .operators import PhysicalOperatorLayer

N_OPS = 4  # dong, gang, cu, ju


class PhysicalPipeline(nn.Module):
    """
    物理流水线

    RGB [B,3,H,W]
      ↓ PhysicalOperatorLayer (4算子)
    4 响应图 [B,4,H,W]
      ↓ 1×1 conv 投影 (1→4 per operator)
    4×4 特征 [B,4,4,H,W]
      ↓ reshape
    16 维本质场 [B,16,H,W]
    """

    def __init__(self):
        super().__init__()
        self.operator_layer = PhysicalOperatorLayer()
        self.projections = nn.ModuleDict({
            f'op{i}': nn.Conv2d(1, N_OPS, 1, bias=False)
            for i in range(N_OPS)
        })
        for proj in self.projections.values():
            nn.init.uniform_(proj.weight, 0.0, 0.5)
            with torch.no_grad():
                for i in range(min(proj.weight.shape[0], proj.weight.shape[1])):
                    proj.weight[i, i] = 1.0

    def forward(self, x):
        base = self.operator_layer(x)  # [B, 4, H, W]

        multi = []
        for i in range(N_OPS):
            ch = base[:, i:i+1, :, :]
            feat = self.projections[f'op{i}'](ch)
            multi.append(feat)

        field = torch.cat([m.unsqueeze(1) for m in multi], dim=1)
        B, n_ops, d, H, W = field.shape
        return field.reshape(B, n_ops * d, H, W)

    def clamp_weights(self):
        for proj in self.projections.values():
            proj.weight.data.clamp_(min=0.0)
