"""
物理算子流水线 v4 — 13算子 + 空间坐标 → 本质场

RGB → PhysicalOperatorLayer (13ch: 8物理+虚空+4局部统计)
  ↓ 8主算子 → 投影(1→8) → BilinearFusion(8×8=64d)
  ↓ + 4局部统计 + void_prob (=5d) + 2空间坐标 (=2d)
  ↓ = 71维本质场
"""

import torch
import torch.nn as nn
from .operators import PhysicalOperatorLayer

N_BASE_OPS = 8          # 参与双线性融合的主算子数
N_AUX = 5               # 辅助通道：4局部统计 + void_prob
N_FULL = 71             # 64 + 5 + 2


class BilinearFusion(nn.Module):
    """双投影融合: W_up/W_dn 混合N个算子 → N×N 交互维"""

    def __init__(self, n_ops=N_BASE_OPS):
        super().__init__()
        self.W_up = nn.Parameter(torch.ones(n_ops, n_ops))
        self.W_dn = nn.Parameter(torch.ones(n_ops, n_ops))

    def forward(self, F):
        B, n_ops, d, H, W = F.shape
        Fu = torch.einsum('bnihw,km->bnmhw', F, self.W_up)
        Fd = torch.einsum('bnihw,km->bnmhw', F, self.W_dn)
        Fu = Fu.permute(0, 3, 4, 1, 2)
        Fd = Fd.permute(0, 3, 4, 1, 2)
        interact = torch.matmul(Fu, Fd.transpose(-1, -2))
        return interact.reshape(B, H, W, n_ops * n_ops).permute(0, 3, 1, 2)

    def clamp_weights(self):
        self.W_up.data.clamp_(min=0.0)
        self.W_dn.data.clamp_(min=0.0)


class PhysicalPipeline(nn.Module):
    """
    PhysicalPipeline v4

    RGB [B,3,H,W]
      ↓ PhysicalOperatorLayer
    13 响应图 [B,13,H,W]   (8主算子+4局部统计+void_prob)
      ↓ 取前8主算子 → 实例归一化
      ↓ 1×1投影(1→8 each)
    8×8 基 [B,8,8,H,W]
      ↓ BilinearFusion (W_up/W_dn)
    64 交互场 [B,64,H,W]
      ↓ + aux[5] + coord[2] = 71维本质场
    """

    def __init__(self):
        super().__init__()
        self.operator_layer = PhysicalOperatorLayer()
        # 8主算子的1×1投影 (1→8)
        self.projections = nn.ModuleDict({
            f'op{i}': nn.Conv2d(1, N_BASE_OPS, 1, bias=False)
            for i in range(N_BASE_OPS)
        })
        self.fusion = BilinearFusion(N_BASE_OPS)
        # 恒等初始化投影
        for proj in self.projections.values():
            w = torch.zeros(N_BASE_OPS, 1, 1, 1)
            w[0] = 1.0
            for j in range(1, N_BASE_OPS):
                w[j] = 0.1
            proj.weight.data = w

    def _coord_channels(self, B, H, W, device):
        yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([yy, xx], dim=1)

    def forward(self, x):
        base = self.operator_layer(x)  # [B, 13, H, W]

        # ─── 8主算子 → 投影 → 双线性融合 ───
        main = base[:, :N_BASE_OPS, :, :]  # [B, 8, H, W]
        main = torch.nn.functional.instance_norm(main)

        multi = []
        for i in range(N_BASE_OPS):
            ch = main[:, i:i+1, :, :]
            feat = self.projections[f'op{i}'](ch)
            multi.append(feat)
        F = torch.stack(multi, dim=1)  # [B, 8, 8, H, W]
        field = self.fusion(F)  # [B, 64, H, W]

        # ─── 辅助通道直接拼接 ───
        aux = base[:, N_BASE_OPS:, :, :]  # [B, 5, H, W] (4局部统计+void_prob)

        # ─── 空间坐标 ───
        B, C, H, W = field.shape
        coord = self._coord_channels(B, H, W, field.device)

        return torch.cat([field, aux, coord], dim=1)  # [B, 71, H, W]

    def clamp_weights(self):
        for proj in self.projections.values():
            proj.weight.data.clamp_(min=0.0)
        self.fusion.clamp_weights()
