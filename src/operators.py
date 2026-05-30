"""
物理算子 — 8个可验证物理量测量器

依赖链（从像素可测 → 需要推理）：
  动(梯度) ── 最底层，只依赖RGB差分
  静(无变化) ── 动的互补
    │
    ├→ 刚(硬边界) ── 需要动定位边缘，大梯度+窄过渡带
    ├→ 柔(软纹理) ── 需要动检测纹理密度，小梯度+宽分布
    │     │
    │     ├→ 聚(围合) ── 需要刚判定"被硬边界包围"
    │     ├→ 散(开放) ── 不被包围的暴露区域
    │           │
    │           ├→ 阳(实体) ── 需要聚的产物：在物体内部
    │           ├→ 阴(虚空) ── 聚的补集：在物体外部

设计约束：
  - 所有算子输出非负 [B, 1, H, W]
  - 下层算子不依赖上层（动不读刚）
  - 每个算子只测一种物理量，不混合语义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _box_filter(x, k=5):
    """Box 均值滤波，空间平滑（replicate 填充，边界外延不引入零）"""
    w = torch.ones(1, 1, k, k, device=x.device) / (k * k)
    p = k // 2
    x_pad = F.pad(x, (p, p, p, p), mode='replicate')
    return F.conv2d(x_pad, w, padding=0)


def _sobel_magnitude(x):
    """
    Sobel 梯度幅值，三通道分别算后取 L2 范数。
    输入 [B, 3, H, W]，输出 [B, 1, H, W]。
    这是动算子的物理基础——RGB 空间变化量。
    """
    device = x.device
    # Sobel 核
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3)

    # replicate padding: 边界像素外延，不创造对称抵消
    x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(x_pad, kx.repeat(3, 1, 1, 1), padding=0, groups=3)  # [B,3,H,W]
    gy = F.conv2d(x_pad, ky.repeat(3, 1, 1, 1), padding=0, groups=3)

    mag = torch.sqrt(gx ** 2 + gy ** 2)  # [B,3,H,W]
    # 三通道 L2 范数 → 单通道
    mag = torch.norm(mag, dim=1, keepdim=True)  # [B,1,H,W]
    return mag


# Sobel 3×3 后 box_filter 抹平的实用最大响应
# 理论 max = sqrt(48) ≈ 6.928（三通道全跳变）
# box_filter(k=3) 抹平后 ≈ 4.62
# 用 4.0 确保锐边 dong≈0.94，渐变 dong≈0.016
_MAX_SOBEL = 4.0


def _normalize_by_max(x):
    """固定除数归一化：保留连续强度，弱梯度不消失"""
    return torch.clamp(x / _MAX_SOBEL, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# 第1层：像素直接可测
# ═══════════════════════════════════════════════════════════════

def dong(x):
    """
    动 — 像素级变化强度（梯度幅值）

    物理量：该位置 RGB 值在空间上的变化速率。
    只依赖 RGB 差分，不依赖任何其他算子。

    计算：
      1. 三通道 Sobel 梯度
      2. 三通道 L2 范数 → 单通道梯度幅值
      3. Box 平滑去噪
      4. /amax 归一化

    输入: [B, 3, H, W] RGB 图像
    输出: [B, 1, H, W] 非负梯度幅值，[0, 1]

    高值区：边缘、纹理、噪声
    低值区：平坦单色区域
    """
    mag = _sobel_magnitude(x)           # [B,1,H,W] 绝对梯度值
    mag = _box_filter(mag, k=3)         # 去噪
    return _normalize_by_max(mag)       # /6.928 保留强度量纲


def jing(x):
    """
    静 — 无变化区域（动的互补）

    物理量：该像素邻域内 RGB 的一致性。
    jing = 1 / (1 + dong * scale)

    输入: [B, 3, H, W] RGB 图像
    输出: [B, 1, H, W] 非负，[0, 1]

    高值区：平坦背景、均匀表面
    低值区：边缘、纹理密集区
    """
    d = dong(x)                         # 动响应
    j = 1.0 / (1.0 + d * 10.0)         # 动越大 → 静越小
    return j


# ═══════════════════════════════════════════════════════════════
# 注册表
# ═══════════════════════════════════════════════════════════════

PHYSICAL_OPERATORS = {
    'dong': dong,  # 动=变化
    'jing': jing,  # 静=恒定
    # 以下算子依赖动/静，待实现：
    # 'gang': gang,  # 刚=硬边界 (需动)
    # 'rou':  rou,   # 柔=软纹理 (需动+静)
    # 'ju':   ju,    # 聚=围合   (需刚)
    # 'san':  san,   # 散=开放   (需刚+柔)
    # 'yang': yang,  # 阳=实体   (需聚)
    # 'yin':  yin,   # 阴=虚空   (需聚)
}


class PhysicalOperatorLayer(nn.Module):
    """
    8 物理算子层
    输入 RGB → 输出 8 通道响应 [B, 8, H, W]
    当前已实现：动、静（2/8）
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        maps = []
        for name in ['dong', 'jing']:  # 按依赖序
            out = PHYSICAL_OPERATORS[name](x)
            maps.append(out.squeeze(1))  # [B,H,W]
        # 未实现的算子填零占位
        B, _, H, W = x.shape
        for _ in range(len(maps), 8):
            maps.append(torch.zeros(B, H, W, device=x.device))
        return torch.stack(maps, dim=1)  # [B,8,H,W]
