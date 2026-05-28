"""
八卦算子 — 颜色原生版 [VERSION=native]

每个算子直接接收 RGB [B, 3, H, W]，内部所有计算基于颜色向量空间。
梯度、方差、曲率、对比度都使用 RGB 三通道信息，
输出仍为单通道响应图 [B, 1, H, W]。

算子固定，无可训练参数。投影层和 A 核不需要修改。
"""

import torch
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════
# 颜 色 感 知 基 础 函 数
# ═══════════════════════════════════════════════════════════

def _sobel_kernels(device, dtype):
    """返回 (sobel_x, sobel_y) 的 4D 核 [1,1,3,3]"""
    sx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                      dtype=dtype, device=device)
    sy = sx.transpose(2, 3)
    return sx, sy


def _color_gradient(x):
    """
    颜色梯度幅值
    对每个通道独立算 Sobel 梯度，再合并为总梯度幅值
    输入 [B,3,H,W]  输出 [B,1,H,W]
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    sx, sy = _sobel_kernels(device, dtype)
    # 扩展到每个通道的深度可分离卷积
    sx_3 = sx.expand(C, -1, -1, -1).contiguous()
    sy_3 = sy.expand(C, -1, -1, -1).contiguous()
    gx = F.conv2d(x, sx_3, padding=1, groups=C)  # [B, C, H, W]
    gy = F.conv2d(x, sy_3, padding=1, groups=C)
    # 所有通道的梯度幅值求和
    gm = torch.sqrt((gx**2 + gy**2).sum(dim=1, keepdim=True) + 1e-6)
    return gm  # [B, 1, H, W]


def _color_gradient_4dir(x):
    """
    四方向颜色梯度幅值
    返回 [B, 4, H, W] 四个方向的颜色梯度
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    kernels = [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],   # 0° 水平
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],   # 45°
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],   # 90° 垂直
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],   # 135°
    ]
    grads = []
    for k_def in kernels:
        k = torch.tensor([[k_def]], dtype=dtype, device=device)
        k_c = k.expand(C, -1, -1, -1).contiguous()
        g = F.conv2d(x, k_c, padding=1, groups=C)  # [B, C, H, W]
        gm = torch.sqrt((g**2).sum(dim=1, keepdim=True) + 1e-6)  # [B, 1, H, W]
        grads.append(gm)
    return torch.cat(grads, dim=1)  # [B, 4, H, W]


def _color_variance_local(x, k=5):
    """
    局部颜色方差（协方差矩阵的迹 = 三通道方差之和）
    输入 [B,3,H,W]  输出 [B,1,H,W]
    """
    pd = k // 2
    box = torch.ones(1, 1, k, k, device=x.device) / (k*k)
    # 对每个通道独立做 box filter（手动展开避免 groups 核形状问题）
    mu_list = []
    sq_list = []
    for c in range(3):
        ch = x[:, c:c+1, :, :]
        mu_c = F.conv2d(ch, box, padding=pd)
        sq_c = F.conv2d(ch**2, box, padding=pd)
        mu_list.append(mu_c)
        sq_list.append(sq_c)
    mu = torch.cat(mu_list, dim=1)   # [B, 3, H, W]
    sq = torch.cat(sq_list, dim=1)   # [B, 3, H, W]
    var = (sq - mu**2).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    return var.clamp(min=0)


def _color_laplacian(x):
    """
    颜色拉普拉斯幅值
    对每个通道算拉普拉斯，取幅值之和
    输入 [B,3,H,W]  输出 [B,1,H,W]
    """
    k = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]],
                     dtype=x.dtype, device=x.device)
    lap_list = []
    for c in range(3):
        ch = x[:, c:c+1, :, :]
        lap_list.append(F.conv2d(ch, k, padding=1))
    lap = torch.cat(lap_list, dim=1)  # [B, 3, H, W]
    return torch.abs(lap).sum(dim=1, keepdim=True)  # [B, 1, H, W]


def _color_center_surround(x, inner_k=5, outer_k=15):
    """
    中心-环绕颜色对比度
    中心邻域颜色均值 vs 环绕邻域颜色均值的欧氏距离
    输入 [B,3,H,W]  输出 [B,1,H,W]
    """
    box_in = torch.ones(1, 1, inner_k, inner_k, device=x.device) / (inner_k**2)
    box_out = torch.ones(1, 1, outer_k, outer_k, device=x.device) / (outer_k**2)
    pd_in = inner_k // 2
    pd_out = outer_k // 2
    mu_in_list = []
    mu_out_list = []
    for c in range(3):
        ch = x[:, c:c+1, :, :]
        mu_in_list.append(F.conv2d(ch, box_in, padding=pd_in))
        mu_out_list.append(F.conv2d(ch, box_out, padding=pd_out))
    mu_in = torch.cat(mu_in_list, dim=1)
    mu_out = torch.cat(mu_out_list, dim=1)
    diff = torch.sqrt(((mu_in - mu_out)**2).sum(dim=1, keepdim=True) + 1e-6)
    return diff


# ═══════════════════════════════════════════════════════════
# 八 卦 算 子 — 颜 色 原 生 版
# ═══════════════════════════════════════════════════════════

def _qian(x):
    """
    乾 — 圆对称性
    沿圆环采样点的颜色方差：颜色一致性高 = 强圆对称
    """
    B, C, H, W = x.shape
    device = x.device
    # 能量掩码：防止均匀区域误报
    gm = _color_gradient(x)
    mask = torch.sigmoid((gm - 0.03) * 200)

    out = []
    for r in [2, 4, 6]:
        ang = torch.linspace(0, 2*np.pi, 12, device=device)
        oy = (r * torch.sin(ang)).round().long()
        ox = (r * torch.cos(ang)).round().long()
        pd = r + 2
        pad = F.pad(x, [pd]*4, mode='reflect')
        # 收集圆环上的 12 个颜色样本
        samples = []
        for dy, dx in zip(oy, ox):
            s = torch.roll(pad, (dy.item(), dx.item()), dims=(2, 3))
            s = s[:, :, pd:pd+H, pd:pd+W]
            samples.append(s)  # 每个 [B, 3, H, W]
        samples = torch.stack(samples, dim=1)  # [B, 12, 3, H, W]
        # 计算 12 个颜色向量的总方差
        samples_flat = samples.permute(0, 3, 4, 1, 2)  # [B, H, W, 12, 3]
        mean_c = samples_flat.mean(dim=3, keepdim=True)  # [B, H, W, 1, 3]
        var_c = ((samples_flat - mean_c)**2).sum(dim=-1).mean(dim=-1)  # [B, H, W]
        var_map = var_c.unsqueeze(1)  # [B, 1, H, W]
        out.append(torch.exp(-var_map * 10))
    return torch.stack(out, dim=0).mean(dim=0) * mask


def _kun(x):
    """
    坤 — 平坦度
    局部颜色方差 / 全局颜色方差
    """
    local_var = _color_variance_local(x, k=9)
    global_var = x.var(dim=[2, 3], keepdim=True).sum(dim=1, keepdim=True) + 1e-6
    return torch.exp(-local_var / global_var)


def _zhen(x):
    """
    震 — 边缘强度
    颜色拉普拉斯幅值（检测颜色突变）
    """
    lap = _color_laplacian(x)
    return torch.tanh(lap * 5)


def _xun(x):
    """
    巽 — 细长纹理/方向性散布
    四方向颜色梯度的一致性：低一致性 = 单一方向纹理
    """
    grads = _color_gradient_4dir(x)  # [B, 4, H, W]
    # 抑制均匀区域
    gm = _color_gradient(x)
    energy_mask = torch.sigmoid((gm - 0.03) * 200)
    # 方向一致性：四方向梯度方差低 = 单一方向占优
    g_mean = grads.mean(dim=1, keepdim=True)
    g_var = ((grads - g_mean)**2).mean(dim=1, keepdim=True)
    thinness = torch.exp(-g_var * 5)
    return thinness * energy_mask


def _kan(x):
    """
    坎 — 曲率
    颜色梯度场的曲率 = div(∇g/|∇g|)，g 为颜色梯度幅值
    """
    gm = _color_gradient(x)  # [B, 1, H, W] — 颜色梯度标量场
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    gx = F.conv2d(gm, sobel, padding=1)
    gy = F.conv2d(gm, sobel.transpose(2, 3), padding=1)
    g_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
    nx = gx / (g_mag + 1e-6)
    ny = gy / (g_mag + 1e-6)
    nxx = F.conv2d(nx, sobel, padding=1)
    nyy = F.conv2d(ny, sobel.transpose(2, 3), padding=1)
    curvature = torch.abs(nxx + nyy)
    return torch.tanh(curvature * 10)


def _li(x):
    """
    离 — 视觉能量
    颜色梯度幅值 + 色彩饱和度 + 亮度
    """
    ge = torch.tanh(_color_gradient(x) * 5)
    # 色彩饱和度（三通道标准差）
    color_var = x.std(dim=1, keepdim=True)
    sat = torch.tanh(color_var * 5)
    # 亮度（三通道均值）
    bri = torch.sigmoid((x.mean(dim=1, keepdim=True) - 0.3) * 5)
    return ge * 0.4 + sat * 0.3 + bri * 0.3


def _gen(x):
    """
    艮 — 块状/遮挡边界
    局部 patch 的颜色方差变化率
    """
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=device)
    # 对每个通道算 patch 方差
    var_maps = []
    ps, pd = 15, 7
    for c in range(3):
        ch = x[:, c:c+1, :, :]
        _, _, Hc, Wc = ch.shape
        padded = F.pad(ch, [pd]*4, mode='reflect')
        patches = F.unfold(padded, kernel_size=ps, stride=1)
        pv = patches.var(dim=1, unbiased=False).view(-1, 1, Hc, Wc)
        var_maps.append(pv)
    # 三通道方差之和 = 颜色 patch 方差
    total_var = torch.cat(var_maps, dim=1).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    blockiness = torch.tanh(total_var * 20)
    # 检测方差变化率 = 遮挡边界
    gx = F.conv2d(total_var, sobel, padding=1)
    gy = F.conv2d(total_var, sobel.transpose(2, 3), padding=1)
    boundary = torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 10)
    return torch.max(blockiness * 0.5, boundary * 0.8)


def _dui(x):
    """
    兑 — 凹陷/缺损
    中心-环绕颜色对比度
    """
    diff = _color_center_surround(x, inner_k=5, outer_k=15)
    # 能量门限
    gm = _color_gradient(x)
    energy_mask = torch.sigmoid((gm - 0.03) * 200)
    concavity = torch.sigmoid((diff - 0.02) * 40) * energy_mask
    # 对比度不对称性 = 开口检测
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    gx = F.conv2d(diff, sobel, padding=1)
    gy = F.conv2d(diff, sobel.transpose(2, 3), padding=1)
    asymmetry = torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 10) * energy_mask
    return torch.max(concavity, asymmetry)


# ═══════════════════════════════════════════════════════════
# 算 子 注 册
# ═══════════════════════════════════════════════════════════

BAGUA_OPERATORS = {
    "qian": ("乾天", _qian),
    "kun": ("坤地", _kun),
    "zhen": ("震雷", _zhen),
    "xun": ("巽风", _xun),
    "kan": ("坎水", _kan),
    "li": ("离火", _li),
    "gen": ("艮山", _gen),
    "dui": ("兑泽", _dui),
}

BAGUA_NAMES = {k: v[0] for k, v in BAGUA_OPERATORS.items()}


class BaguaOperatorLayer:
    """颜色原生算子层"""
    def __call__(self, x):
        return {name: fn(x) for name, (_, fn) in BAGUA_OPERATORS.items()}


if __name__ == "__main__":
    import cv2 as cv
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    cv.circle(img, (112, 112), 70, (60, 60, 60), -1)
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    layer = BaguaOperatorLayer()
    r = layer(x)
    print("颜色原生算子测试：")
    for name, (cn, _) in BAGUA_OPERATORS.items():
        v = r[name][0, 0, 100:124, 100:124].mean().item()
        print(f"  {cn:6s}: {v:.4f}")
