"""
本质场自组织扩散

物理直觉：场内同物质像素的 64 维向量互相吸引 → 逐步靠拢。
边缘处扩散被阻断 → 物质边界自然浮现。

一轮迭代信息传播半径 = kernel_size/2。
N 轮后全局结构编码到每个像素。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldDiffusion(nn.Module):
    """
    本质场自组织扩散

    Args:
        tau: 温度参数。越小 → 只聚合非常相似的向量 → 划分更细
        alpha: 扩散步长。越大 → 收敛快但可能振荡
        n_iters: 迭代次数。3-10，太少不收敛，太多过度平滑
        kernel_size: 邻域窗口大小。对应感受野
        edge_scale: 边缘门强度。越大 → 边缘阻断越强
    """

    def __init__(self, tau=0.1, alpha=0.3, n_iters=5,
                 kernel_size=9, edge_scale=3.0):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.n_iters = n_iters
        self.kernel_size = kernel_size
        self.edge_scale = edge_scale
        # 可学习温度（初始化为传入值）
        self.log_tau = nn.Parameter(torch.tensor(tau).log())

    def _compute_edge_gate(self, operator_maps):
        """
        从算子响应计算边缘门。
        震(zhen) = 拉普拉斯，离(li) = 梯度幅值。
        两者结合 → 边缘强度。

        gate: 边缘处≈0（阻断扩散），内部≈1（自由扩散）
        """
        # operator_maps 中的值范围因算子而异，先各自归一化
        edge_zhen = operator_maps['zhen']  # [B, 1, H, W]
        edge_li = operator_maps['li']      # [B, 1, H, W]

        # 稳健归一化：2/98 百分位
        def robust_norm(x):
            B, C, H, W = x.shape
            x_flat = x.view(B, -1)
            lo = x_flat.kthvalue(max(1, int(0.02 * x_flat.shape[1])), dim=1,
                                 keepdim=True).values.view(B, 1, 1, 1)
            hi = x_flat.kthvalue(min(x_flat.shape[1], int(0.98 * x_flat.shape[1])), dim=1,
                                 keepdim=True).values.view(B, 1, 1, 1)
            return torch.clamp((x - lo) / (hi - lo + 1e-8), 0, 1)

        e_z = robust_norm(edge_zhen)
        e_l = robust_norm(edge_li)
        edge = torch.max(e_z, e_l)  # 取较强信号

        # 软门：高边缘 → 低门值
        gate = torch.exp(-edge * self.edge_scale)
        return gate  # [B, 1, H, W]

    def _diffusion_step(self, phi, edge_gate):
        """
        单步扩散：
        1. unfold 提取邻域 patch
        2. 计算中心像素与邻域的向量相似度
        3. 相似度加权平均 → 新向量
        4. 边缘门混合：边缘处保留原始向量
        """
        B, C, H, W = phi.shape
        k = self.kernel_size
        pad = k // 2
        tau = self.log_tau.exp()

        # unfold → [B, C*k*k, H*W]
        phi_unfold = F.unfold(phi, kernel_size=k, padding=pad)
        phi_unfold = phi_unfold.view(B, C, k * k, H, W)  # [B, C, K², H, W]

        # 中心向量 [B, C, 1, H, W]
        phi_center = phi.unsqueeze(2)

        # L2 距离 → 相似度
        diff = phi_unfold - phi_center                      # [B, C, K², H, W]
        dist_sq = (diff * diff).sum(dim=1)                   # [B, K², H, W]
        sim = torch.exp(-dist_sq / (tau + 1e-8))             # [B, K², H, W]

        # 归一化权重
        sim = sim / (sim.sum(dim=1, keepdim=True) + 1e-8)

        # 加权平均邻域向量
        phi_new = (phi_unfold * sim.unsqueeze(1)).sum(dim=2)  # [B, C, H, W]

        # 边缘门混合：边缘处保留原始，内部用扩散值
        phi_new = phi_new * edge_gate + phi * (1 - edge_gate)

        # 步长更新
        phi_out = (1 - self.alpha) * phi + self.alpha * phi_new

        return phi_out

    def forward(self, field_64, operator_maps):
        """
        Args:
            field_64: [B, 64, H, W] 本质场
            operator_maps: Dict[name -> [B, 1, H, W]] 原始算子响应（用于边缘门）

        Returns:
            field_diffused: [B, 64, H, W] 扩散后本质场
            convergence: list[float] 每轮迭代的向量变化量（用于诊断）
        """
        phi = field_64
        edge_gate = self._compute_edge_gate(operator_maps)

        convergence = []
        for it in range(self.n_iters):
            phi_new = self._diffusion_step(phi, edge_gate)
            delta = (phi_new - phi).norm() / (phi.norm() + 1e-8)
            convergence.append(delta.item())
            phi = phi_new

        return phi, convergence


def bilateral_diffusion(field_64, operator_maps, n_iters=5,
                        spatial_sigma=3.0, range_sigma=0.1):
    """
    双边滤波风格的扩散（纯函数，无可学习参数）。
    同时考虑空间距离和向量相似度。

    比 FieldDiffusion 更慢但边缘保持更好。
    适用于小图/离线分析。
    """
    B, C, H, W = field_64.shape

    # 计算边缘门
    edge_zhen = operator_maps['zhen']
    edge_li = operator_maps['li']
    edge = torch.max(
        edge_zhen / (edge_zhen.max() + 1e-8),
        edge_li / (edge_li.max() + 1e-8)
    )
    edge_gate = torch.exp(-edge * 3.0)

    # 预计算空间权重（各向同性高斯）
    ks = int(spatial_sigma * 3) * 2 + 1
    ys = torch.arange(ks, device=field_64.device).float() - ks // 2
    xs = torch.arange(ks, device=field_64.device).float() - ks // 2
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    spatial_w = torch.exp(-(gy**2 + gx**2) / (2 * spatial_sigma**2))
    spatial_w = spatial_w.view(1, 1, ks, ks, 1)  # [1, 1, K, K, 1]

    phi = field_64
    convergence = []

    for _ in range(n_iters):
        phi_unfold = F.unfold(phi, kernel_size=ks, padding=ks // 2)
        phi_unfold = phi_unfold.view(B, C, ks, ks, H, W)

        phi_center = phi.unsqueeze(2).unsqueeze(2)
        diff = phi_unfold - phi_center
        dist_sq = (diff * diff).sum(dim=1, keepdim=True)
        range_w = torch.exp(-dist_sq / (2 * range_sigma**2))

        weights = spatial_w * range_w  # [B, 1, K, K, H, W]
        weights = weights / (weights.sum(dim=(2, 3), keepdim=True) + 1e-8)

        phi_new = (phi_unfold * weights).sum(dim=(2, 3))
        phi_new = phi_new * edge_gate + phi * (1 - edge_gate)

        delta = (phi_new - phi).norm() / (phi.norm() + 1e-8)
        convergence.append(delta.item())
        phi = phi_new

    return phi, convergence
