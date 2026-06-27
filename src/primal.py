"""
Primal Field v0 — 最小基元，单一规则

设计哲学：
  像素的本性 = 相邻像素互相影响，影响强度 = 它们的相似度
  不定义算子。不定义K。不定义边界。

  相似 → 吸引 → 靠拢
  相异 → 不吸引 → 自然分界

  演化后: 向量一致的连通区域 = 物质
"""

import torch
import numpy as np


def primal_relax(field, tau=0.05, alpha=0.3, n_iters=50, repulsion=0.0):
    """
    场弛豫。规则：吸引 + 自适应排斥。

    自适应: 每个像素自己算邻域争议度。
           争议大→强排斥(边界/缝隙)，争议小→弱排斥(平坦区)。

    repulsion 参数是排斥力物理上限，非全局强度。
    """
    B, C, H, W = field.shape
    phi = field.clone()
    conv = []

    for _ in range(n_iters):
        prev = phi.clone()

        attract = torch.zeros_like(phi)
        awsum = torch.zeros(B, 1, H, W, device=field.device)
        repel = torch.zeros_like(phi)
        rwsum = torch.zeros(B, 1, H, W, device=field.device)
        all_sim = torch.zeros(B, 4, H, W, device=field.device)

        for i, (dy, dx) in enumerate([(0, 1), (0, -1), (1, 0), (-1, 0)]):
            nb = torch.roll(phi, shifts=(dy, dx), dims=(2, 3))
            diff = (nb - phi).pow(2).sum(dim=1, keepdim=True)
            sim = torch.exp(-diff / tau)

            attract += nb * sim
            awsum += sim

            # 记录每个方向的相似度（用于自适应排斥）
            all_sim[:, i] = sim[:, 0]

            if repulsion > 0:
                push = (phi - nb) * (1.0 - sim)
                repel += push
                rwsum += (1.0 - sim)

        phi_attract = attract / (awsum + 1e-8)

        if repulsion > 0:
            # 自适应排斥强度: 邻域争议度 = 1 - 平均相似度
            consensus = all_sim.mean(dim=1, keepdim=True)   # [B,1,H,W]
            controversy = 1.0 - consensus                     # [B,1,H,W]
            adaptive_rep = controversy * repulsion            # 上限 × 争议度
            phi_repel = phi + repel / (rwsum + 1e-8)
            phi_new = phi_attract * (1 - adaptive_rep) + phi_repel * adaptive_rep
        else:
            phi_new = phi_attract

        phi = (1 - alpha) * phi + alpha * phi_new

        d = (phi - prev).norm() / (phi.norm() + 1e-8)
        conv.append(d.item())
        if d < 1e-4:
            break

    return phi, conv


def extract_domains(field_relaxed, grad_pct=75, min_domain_size=None):
    """
    从弛豫场中提取物质域。

    场梯度低 = 邻域一致 = 物质内部。连通分量 = 物质域。

    Args:
        field_relaxed: [B, C, H, W]
        grad_pct: 梯度阈值百分位
        min_domain_size: 最小域大小（像素数），None=1%面积

    Returns:
        labels: [H, W] 物质标签
        n_domains: int
    """
    B, C, H, W = field_relaxed.shape
    field_np = field_relaxed[0].detach().cpu().numpy()

    # 场梯度
    gy = np.abs(np.diff(field_np, axis=1, append=field_np[:, -1:, :])).mean(0)
    gx = np.abs(np.diff(field_np, axis=2, append=field_np[:, :, -1:])).mean(0)
    grad = gy + gx

    # 梯度低 = 物质内部
    interior = grad < np.percentile(grad, grad_pct)

    from scipy import ndimage
    labels, n_raw = ndimage.label(interior)

    # 最小尺寸过滤
    if min_domain_size is None:
        min_domain_size = H * W // 100
    if min_domain_size < 50:
        min_domain_size = 50

    sizes = ndimage.sum(np.ones_like(labels), labels,
                        index=range(1, n_raw + 1))
    new_labels = np.zeros_like(labels)
    next_id = 1
    for lid in range(1, n_raw + 1):
        if sizes[lid - 1] >= min_domain_size:
            new_labels[labels == lid] = next_id
            next_id += 1

    labels = new_labels - 1
    labels = labels.clip(min=0)
    n_domains = labels.max() + 1 if labels.max() >= 0 else 0

    return labels, n_domains
