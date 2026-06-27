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


def enrich_field(rgb_field):
    """
    RGB(3) → RGB + 结构关系(8维)。

    4方向邻域差异: 每个方向 = RGB 三通道差的绝对值均值。
    R_diff≈G_diff≈B_diff → 3→1 压缩。
    局部方差: 5x5 窗口。

    人不选择维度。信息源完备。
    """
    B, C, H, W = rgb_field.shape
    device = rgb_field.device

    parts = [rgb_field]

    # 4方向结构关系
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nb = torch.roll(rgb_field, shifts=(dy, dx), dims=(2, 3))
        diff = (rgb_field - nb).abs().mean(dim=1, keepdim=True)
        parts.append(diff)

    # 局部方差
    kernel = torch.ones(1, 1, 5, 5, device=device) / 25
    gray = rgb_field.mean(dim=1, keepdim=True)
    local_mean = torch.nn.functional.conv2d(gray, kernel, padding=2)
    local_sq_mean = torch.nn.functional.conv2d(gray * gray, kernel, padding=2)
    local_var = (local_sq_mean - local_mean * local_mean).clamp(min=0)
    parts.append(local_var)

    return torch.cat(parts, dim=1)


def primal_relax(field, alpha=0.3, n_iters=50, repulsion=0.0,
                 inertia=False):
    """
    场弛豫。tau 每像素自算——零全局常数。
    """
    B, C, H, W = field.shape
    phi = field.clone()
    conv = []
    stability = torch.zeros(B, 1, H, W, device=field.device) if inertia else None

    # 每像素的自身性质（从初始场计算，不随演化改变）
    diffs_init = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nb = torch.roll(field, shifts=(dy, dx), dims=(2, 3))
        d = (nb - field).pow(2).sum(dim=1, keepdim=True)
        diffs_init.append(d)
    diffs_stack = torch.stack(diffs_init)  # [4, B, 1, H, W]
    tau_map = diffs_stack.median(dim=0)[0].clamp(min=1e-6)      # tau: 中位数距离
    grad_map = diffs_stack.mean(dim=0)                            # 局部梯度 = 平均距离

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
            # 每像素自己的 tau —— 零全局常数
            sim = torch.exp(-diff / tau_map)

            attract += nb * sim
            awsum += sim

            # 记录每个方向的相似度（用于自适应排斥）
            all_sim[:, i] = sim[:, 0]

            if repulsion:
                push = (phi - nb) * (1.0 - sim)
                repel += push
                rwsum += (1.0 - sim)

        phi_attract = attract / (awsum + 1e-8)

        if repulsion:
            # 每像素自己的耦合强度 = 自身局部梯度
            # 高梯度 → 需要强排斥（边缘/纹理）。低梯度 → 弱排斥（平坦区）
            consensus = all_sim.mean(dim=1, keepdim=True)
            coupling = grad_map * 0.5  # 自身梯度 → 排斥力尺度
            adaptive_rep = (1.0 - consensus) * coupling
            phi_repel = phi + repel / (rwsum + 1e-8)
            phi_new = phi_attract * (1 - adaptive_rep) + phi_repel * adaptive_rep
        else:
            phi_new = phi_attract

        # 惯性: 稳定越久 → 步长越小
        if inertia:
            change = (phi_new - phi).abs().mean(dim=1, keepdim=True)
            is_stable = change < 0.001
            stability = torch.where(is_stable, stability + 1,
                                    torch.zeros_like(stability))
            adaptive_alpha = alpha / (1.0 + stability * 0.3)
        else:
            adaptive_alpha = alpha

        phi = (1 - adaptive_alpha) * phi + adaptive_alpha * phi_new

        d = (phi - prev).norm() / (phi.norm() + 1e-8)
        conv.append(d.item())
        if d < 1e-4:
            break

    return phi, conv


def primal_relax_multiscale(field, alpha=0.3, n_iters=50,
                              repulsion=0.0, coarse_weight_max=0.15):
    """
    多尺度弛豫——每像素自适应。

    局部碎片化程度高 → 需要粗尺度引导 → blend 权重大
    局部平坦一致 → 不需要粗尺度 → blend 权重小
    """
    B, C, H, W = field.shape
    phi = field.clone()
    conv = []

    for _ in range(n_iters):
        prev = phi.clone()

        # 细尺度: 原图上的吸引+排斥
        phi, _c = primal_relax(phi, alpha=alpha * 0.5,
                                n_iters=1, repulsion=repulsion * 0.3)

        # 粗尺度: 降采样 → 弛豫 → 上采样
        coarse = torch.nn.functional.interpolate(
            phi, scale_factor=0.5, mode='bilinear')
        coarse, _ = primal_relax(coarse, alpha=alpha * 0.3,
                                  n_iters=1, repulsion=repulsion * 0.7)
        coarse_up = torch.nn.functional.interpolate(
            coarse, size=(H, W), mode='bilinear')

        # 融合权重 = 粗细尺度差异。差异大 → 粗尺度有结构信息
        scale_diff = (phi - coarse_up).abs().mean(dim=1, keepdim=True)
        adaptive_weight = scale_diff.clamp(0, 0.3)

        phi = phi * (1 - adaptive_weight) + coarse_up * adaptive_weight

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
