"""
物理算子流水线 v2 — 4算子 → 双投影融合 → 16维本质空间

RGB → 4物理算子 → 投影(1→4) → W_up/W_dn融合 → 4×4=16维场
"""

import torch
import torch.nn as nn
from .operators import PhysicalOperatorLayer

N_OPS = 4


class BilinearFusion(nn.Module):
    """
    双投影融合 — 算子间交叉交互

    F [B, 4, 4, H, W]: 4算子 × 4投影维
    W_up/W_dn ∈ R^(4×4): 混合权重

    Fu = W_up @ F    (上投影)
    Fd = W_dn @ F    (下投影)
    interact = Fu · Fd^T → 4×4=16维 per pixel
    """

    def __init__(self, d=N_OPS):
        super().__init__()
        self.W_up = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.02)
        self.W_dn = nn.Parameter(torch.eye(d) + torch.randn(d, d) * 0.02)

    def forward(self, F):
        B, n_ops, d, H, W = F.shape
        Fu = torch.einsum('bnihw,km->bnmhw', F, self.W_up)  # [B,4,4,H,W]
        Fd = torch.einsum('bnihw,km->bnmhw', F, self.W_dn)  # [B,4,4,H,W]
        Fu = Fu.permute(0, 3, 4, 1, 2)  # [B,H,W,4,4]
        Fd = Fd.permute(0, 3, 4, 1, 2)
        interact = torch.matmul(Fu, Fd.transpose(-1, -2))  # [B,H,W,4,4]
        return interact.reshape(B, H, W, n_ops * n_ops).permute(0, 3, 1, 2)

    def clamp_weights(self):
        self.W_up.data.clamp_(min=0.0)
        self.W_dn.data.clamp_(min=0.0)


class PhysicalPipeline(nn.Module):
    """
    物理流水线 v2 — 带双投影融合

    RGB [B,3,H,W]
      ↓ PhysicalOperatorLayer
    4 响应图 [B,4,H,W]
      ↓ 1×1 conv 投影 (1→4 per operator)
    4×4 特征 [B,4,4,H,W]
      ↓ BilinearFusion (W_up/W_dn 交叉)
    16 维本质场 [B,16,H,W]
    """

    def __init__(self):
        super().__init__()
        self.operator_layer = PhysicalOperatorLayer()
        self.projections = nn.ModuleDict({
            f'op{i}': nn.Conv2d(1, N_OPS, 1, bias=False)
            for i in range(N_OPS)
        })
        self.fusion = BilinearFusion(d=N_OPS)
        # Init projections with identity + small noise
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
            feat = self.projections[f'op{i}'](ch)  # [B, 4, H, W]
            multi.append(feat)
        F = torch.stack(multi, dim=1)  # [B, 4, 4, H, W]
        return self.fusion(F)  # [B, 16, H, W]

    def clamp_weights(self):
        for proj in self.projections.values():
            proj.weight.data.clamp_(min=0.0)
        self.fusion.clamp_weights()
