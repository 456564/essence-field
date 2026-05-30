"""
物理算子 — 4个独立可测物理量

依赖链：
  RGB → dong(梯度幅值) ─→ gang(梯度脊线)
       ├→ cu(粗糙度)
       └→ ju(围合度, 读gang)

设计约束：
  - 每个算子测一种客观物理量，不存在"X 的补集"
  - 非负输出 [B, 1, H, W]
  - 下层不依赖上层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def _box_filter(x, k=5):
    """Box 均值滤波，replicate 填充。支持任意通道数。"""
    C = x.shape[1]
    w = torch.ones(1, 1, k, k, device=x.device) / (k * k)
    p = k // 2
    x_pad = F.pad(x, (p, p, p, p), mode='replicate')
    return F.conv2d(x_pad, w.repeat(C, 1, 1, 1), padding=0, groups=C)


def _sobel_magnitude(x):
    """Sobel 梯度幅值，三通道 L2 范数 → [B,1,H,W]"""
    device = x.device
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                       device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                       device=device).view(1, 1, 3, 3)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(x_pad, kx.repeat(3, 1, 1, 1), padding=0, groups=3)
    gy = F.conv2d(x_pad, ky.repeat(3, 1, 1, 1), padding=0, groups=3)
    mag = torch.sqrt(gx ** 2 + gy ** 2)  # [B,3,H,W]
    return torch.norm(mag, dim=1, keepdim=True)  # [B,1,H,W]


_MAX_SOBEL = 5.0  # 实用最大 Sobel 响应（box 抹平后）


# ═══════════════════════════════════════════════════════════════
# 算子 1: 动 — 梯度幅值 (L0, RGB直接可测)
# ═══════════════════════════════════════════════════════════════

def dong(x):
    """
    动 — 梯度幅值（变化强度）
    RGB → Sobel → box去噪 → /4.0归一化
    输入 [B,3,H,W] → 输出 [B,1,H,W]，[0,1]
    """
    mag = _sobel_magnitude(x)
    mag = _box_filter(mag, k=3)
    return torch.clamp(mag / _MAX_SOBEL, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# 算子 2: 刚 — 梯度脊线 (L1, 从动)
# ═══════════════════════════════════════════════════════════════

def _gang_from_dong(d):
    """刚 = (dong - box(dong,7)).clamp(min=0)"""
    return (d - _box_filter(d, k=7)).clamp(min=0)


def gang(x):
    """刚 — 梯度脊线（硬边界）。RGB兼容封装。"""
    return _gang_from_dong(dong(x))


# ═══════════════════════════════════════════════════════════════
# 算子 3: 粗 — 局部纹理能量 (L0, RGB直接可测)
# ═══════════════════════════════════════════════════════════════

def cu(x):
    """
    粗 — 局部 RGB 方差（表面粗糙度）

    物理量：邻域内像素 RGB 的均方偏差。
    平滑表面 ≈ 0，粗糙纹理 > 0。
    独立于梯度——边缘处梯度高但纹理可能低。

    cu = box((x - box(x,5))², k=7).mean(channel)

    输入 [B,3,H,W] → 输出 [B,1,H,W]
    """
    local_mean = _box_filter(x, k=5)         # [B,3,H,W] 邻域均值
    residual_sq = (x - local_mean) ** 2       # [B,3,H,W] 偏差平方
    variance = _box_filter(residual_sq, k=7)  # [B,3,H,W] 局部方差
    cu_raw = variance.mean(dim=1, keepdim=True)  # [B,1,H,W] 三通道均值
    return torch.clamp(cu_raw / 0.08, 0.0, 1.0)  # /0.08: 理论max≈0.25, 实用≈0.08


# ═══════════════════════════════════════════════════════════════
# 算子 4: 聚 — 围合度 (L1, 从刚)
# ═══════════════════════════════════════════════════════════════

def _ju_from_gang(g):
    """
    聚 — 被硬边界包围的程度

    物理量：该像素周围 31×31 窗口内刚的密度。
    闭合轮廓内密度高，开放侧密度低。

    ju = box_filter(gang, k=31) / 0.1
    """
    density = _box_filter(g, k=31)
    return torch.clamp(density / 0.05, 0.0, 1.0)


def ju(x):
    """聚 — 围合度。RGB 兼容封装。"""
    return _ju_from_gang(_gang_from_dong(dong(x)))


# ═══════════════════════════════════════════════════════════════
# 注册表
# ═══════════════════════════════════════════════════════════════

PHYSICAL_OPERATORS = {
    'dong': dong,  # 动=梯度幅值 (L0)
    'gang': gang,  # 刚=梯度脊线 (L1←动)
    'cu':   cu,    # 粗=纹理能量 (L0)
    'ju':   ju,    # 聚=围合度   (L1←刚)
}


# ═══════════════════════════════════════════════════════════════
# 算子层
# ═══════════════════════════════════════════════════════════════

class PhysicalOperatorLayer(nn.Module):
    """
    4 物理算子层 — 按依赖链编排

    RGB → dong ─→ gang ─→ ju
         └→ cu
    """

    def forward(self, x):
        d = dong(x)                    # L0
        g = _gang_from_dong(d)         # L1
        c = cu(x)                      # L0 (独立)
        j = _ju_from_gang(g)           # L1

        return torch.cat([d, g, c, j], dim=1)  # [B,4,H,W]
