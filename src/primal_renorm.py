"""
重整化弛豫——共识驱动粒度粗化

物理: 共识形成的区域 → 自然粗化为超像素 → 下阶段以粗粒身份交互
      未共识区 → 保持细粒 → 边界永远精细

不需要人定义K。不需要人定义层级。粒度的粗化是场自己的行为。
"""

import torch
import numpy as np
from .primal import primal_relax, extract_domains


def coarsen_consensus(field, window=5, threshold=0.02):
    """
    共识检测。低方差→已共识→超像素(窗口均值)。
    """
    B, C, H, W = field.shape
    device = field.device

    kernel = torch.ones(1, 1, window, window, device=device) / (window * window)
    field_flat = field.reshape(B * C, 1, H, W)
    local_mean = torch.nn.functional.conv2d(field_flat, kernel, padding=window // 2)
    local_sq = torch.nn.functional.conv2d(field_flat * field_flat, kernel, padding=window // 2)
    local_var = (local_sq - local_mean * local_mean).clamp(min=0)
    local_var = local_var.reshape(B, C, H, W).mean(dim=1, keepdim=True)

    consensus_mask = local_var < threshold
    coarse_val = local_mean.reshape(B, C, H, W)
    phi_coarse = field.clone()
    phi_coarse = phi_coarse * (~consensus_mask).float() + coarse_val * consensus_mask.float()

    return phi_coarse, consensus_mask


def primal_relax_renorm(field, alpha=0.3, n_iters=50, repulsion=False,
                         coarsen_rounds=2):
    """
    重整化弛豫——共识驱动粗化。

    1. 弛豫 N/rounds 轮
    2. 共识检测→合并超像素
    3. 粗场重新弛豫
    4. 重复→粒度逐轮变粗
    """
    phi = field.clone()
    conv = []
    net_force = None
    rounds_per = n_iters // coarsen_rounds

    for cr in range(coarsen_rounds):
        phi, c, nf = primal_relax(phi, alpha=alpha, n_iters=rounds_per,
                                   repulsion=repulsion)
        conv.extend(c)
        net_force = nf

        if cr < coarsen_rounds - 1:
            phi, _ = coarsen_consensus(phi, window=5, threshold=0.005)

    return phi, conv, net_force
