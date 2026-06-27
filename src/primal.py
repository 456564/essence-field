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


def primal_relax(field, tau=0.05, alpha=0.3, n_iters=50,
                 repulsion=0.0):
    """
    场弛豫——吸引+排斥。

    规则: 相似→吸引(拉近)。不相似→排斥(推开)。

    Args:
        field: [B, C, H, W] 初始场向量
        tau: 温度。越小→只吸引极相似邻居
        alpha: 步长
        n_iters: 最大迭代数
        repulsion: 排斥强度 [0, 1)。0=无排斥, 越大越排斥不相似邻居

    Returns:
        field_relaxed, convergence
    """
    B, C, H, W = field.shape
    phi = field.clone()
    conv = []

    for _ in range(n_iters):
        prev = phi.clone()

        # ---- 吸引: 相似邻居拉向我 ----
        attract = torch.zeros_like(phi)
        awsum = torch.zeros(B, 1, H, W, device=field.device)

        # ---- 排斥: 不相似邻居推开我 ----
        repel = torch.zeros_like(phi)
        rwsum = torch.zeros(B, 1, H, W, device=field.device)

        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = torch.roll(phi, shifts=(dy, dx), dims=(2, 3))
            diff = (nb - phi).pow(2).sum(dim=1, keepdim=True)
            sim = torch.exp(-diff / tau)  # [0, 1]

            # 吸引: 相似度 × 邻居向量
            attract += nb * sim
            awsum += sim

            # 排斥: 反向力 = 不相似度 × (我 - 邻居)方向
            #        不相似 → 邻居和我差异大 → 推我远离它
            if repulsion > 0:
                dissimilarity = 1.0 - sim  # [0, 1]
                push = (phi - nb) * dissimilarity  # 推开方向
                repel += push
                rwsum += dissimilarity

        # 吸引: 加权平均
        phi_attract = attract / (awsum + 1e-8)

        # 排斥: 远离不相似邻居
        phi_repel = phi + repel / (rwsum + 1e-8) if repulsion > 0 else phi

        # 合力
        phi_new = phi_attract * (1 - repulsion) + phi_repel * repulsion
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
