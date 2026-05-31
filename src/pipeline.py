"""
物理算子流水线 v3 — 5算子 + 空间坐标 → 本质空间

RGB → 5算子 → 投影(1→N) → BilinearFusion(N×N) + coord → 本质场
"""

import torch
import torch.nn as nn
from .operators import PhysicalOperatorLayer, N_OPS


class BilinearFusion(nn.Module):
    """双投影融合: W_up/W_dn 混合N个算子 → N×N 交互维"""

    def __init__(self, n_ops=N_OPS):
        super().__init__()
        self.W_up = nn.Parameter(torch.eye(n_ops) + torch.randn(n_ops, n_ops) * 0.02)
        self.W_dn = nn.Parameter(torch.eye(n_ops) + torch.randn(n_ops, n_ops) * 0.02)

    def forward(self, F):
        B, n_ops, d, H, W = F.shape
        Fu = torch.einsum('bnihw,km->bnmhw', F, self.W_up)
        Fd = torch.einsum('bnihw,km->bnmhw', F, self.W_dn)
        Fu = Fu.permute(0, 3, 4, 1, 2)  # [B,H,W,N,d]
        Fd = Fd.permute(0, 3, 4, 1, 2)
        interact = torch.matmul(Fu, Fd.transpose(-1, -2))  # [B,H,W,N,N]
        return interact.reshape(B, H, W, n_ops * n_ops).permute(0, 3, 1, 2)

    def clamp_weights(self):
        self.W_up.data.clamp_(min=0.0)
        self.W_dn.data.clamp_(min=0.0)


class PhysicalPipeline(nn.Module):
    """
    物理流水线 v3 — 空间感知

    RGB [B,3,H,W]
      ↓ PhysicalOperatorLayer
    5 响应图 [B,5,H,W]
      ↓ + coord (y, x) → [B,7,H,W]
      ↓ 1×1 conv 投影 (1→N per channel, coord→1 each)
    5×(N) + 2 = 5×5 + 2 = 27 维场
      (or: 5算子→BilinearFusion(25d) + 2coord = 27d)
    """

    def __init__(self, n_ops=N_OPS):
        super().__init__()
        self.n_ops = n_ops
        self.operator_layer = PhysicalOperatorLayer()
        # 5 physical operators → dim projection
        self.projections = nn.ModuleDict({
            f'op{i}': nn.Conv2d(1, n_ops, 1, bias=False)
            for i in range(n_ops)
        })
        # Coord: no projection, keep raw 2 channels
        self.fusion = BilinearFusion(n_ops=n_ops)
        # Init: random orthogonal-ish (break identity, encourage diversity)
        for proj in self.projections.values():
            nn.init.orthogonal_(proj.weight, gain=0.5)

    def _coord_channels(self, B, H, W, device):
        yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([yy, xx], dim=1)  # [B, 2, H, W]

    def forward(self, x):
        base = self.operator_layer(x)  # [B, 5, H, W]

        # Project each physical operator: 1 → n_ops
        multi = []
        for i in range(self.n_ops):
            ch = base[:, i:i+1, :, :]
            feat = self.projections[f'op{i}'](ch)
            multi.append(feat)
        F = torch.stack(multi, dim=1)  # [B, 5, 5, H, W]

        # Bilinear fusion: 5×5 = 25 dim
        field = self.fusion(F)  # [B, 25, H, W]

        # Append spatial coords
        B, C, H, W = field.shape
        coord = self._coord_channels(B, H, W, field.device)
        field = torch.cat([field, coord], dim=1)  # [B, 27, H, W]

        return field

    def clamp_weights(self):
        for proj in self.projections.values():
            proj.weight.data.clamp_(min=0.0)
        self.fusion.clamp_weights()
