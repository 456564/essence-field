"""
八卦算子 — 固定颜色版 [VERSION=fixedcolor]

每个算子天生自带固定的颜色偏好（基于八卦传统颜色），
只对特定颜色通道的形态产生响应。

算子 = 固定颜色权重 × 固定几何检测器
总参数量：0（完全固定）。只有投影层和A核可训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _normalize(v):
    """归一化 RGB 方向向量"""
    n = np.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else v


# ═══════════════════════════════════════════════════════════
# 固 定 颜 色 权 重（八卦传统颜色映射）
# ═══════════════════════════════════════════════════════════

# 每个算子天生只知道"自己该看什么颜色"。
# 权重 = [R, G, B] 系数，正=敏感，负=抑制，0=忽略
FIXED_BAGUA_COLORS = {
    'qian': _normalize([1.0, 0.6, 0.0]),   # 乾→天→赤/玄黄→橙红
    'kun':  _normalize([0.2, 0.5, 0.1]),   # 坤→地→黄/黑→暗绿
    'zhen': _normalize([0.2, 1.0, 0.6]),   # 震→雷→青绿→绿蓝
    'xun':  _normalize([0.9, 0.9, 0.9]),   # 巽→风→白→全色偏亮
    'kan':  _normalize([0.1, 0.3, 1.0]),   # 坎→水→黑/深蓝→蓝
    'li':   [1.0, -0.5, -0.5],            # 离→火→赤→红胜绿蓝
    'gen':  _normalize([0.9, 0.7, 0.2]),   # 艮→山→黄/棕→橙黄
    'dui':  _normalize([0.6, 0.6, 0.9]),   # 兑→泽→白/蓝→泛蓝白
}


# ═══════════════════════════════════════════════════════════
# 灰 度 算 子（纯函数，与灰度版一致）
# ═══════════════════════════════════════════════════════════

def _box_filter(x, k=5):
    w = torch.ones(1, 1, k, k, device=x.device) / (k*k)
    return F.conv2d(x, w, padding=k//2)


def _qian(ch):
    """乾 — 天/圆/完整。环采样一致性=圆度，取各半径最佳。"""
    B, C, H, W = ch.shape
    device = ch.device
    out = []
    for r in [8, 16, 32]:
        ang = torch.linspace(0, 2*np.pi, 12, device=device)
        oy = (r * torch.sin(ang)).round().long()
        ox = (r * torch.cos(ang)).round().long()
        pd = r + 2
        pad = F.pad(ch, [pd]*4, mode='reflect')
        smp = []
        for dy, dx in zip(oy, ox):
            s = torch.roll(pad, (dy.item(), dx.item()), dims=(2,3))
            s = s[:, :, pd:pd+H, pd:pd+W]
            smp.append(s)
        smp = torch.stack(smp, dim=1)
        var_map = smp.var(dim=1, unbiased=False) + 0.01
        out.append(1.0 / var_map)
    return torch.stack(out, dim=0).max(dim=0)[0]


def _kun(ch):
    """坤 — 地/结构平坦。Laplacian密度低=无边缘/纹理=平坦，不管颜色怎么变。"""
    lap_k = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]],
                         dtype=ch.dtype, device=ch.device)
    lap = F.conv2d(ch, lap_k, padding=1).abs()  # [B,1,H,W]
    k = 31
    lap_density = _box_filter(lap, k=k)  # 局部平均边缘强度
    return 1.0 / (lap_density * 5.0 + 1.0)  # 边缘多→坤低, 无边缘→坤高


def _zhen(ch):
    """震 — 颜色拉普拉斯幅值，无压缩"""
    k = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]],
                     dtype=ch.dtype, device=ch.device)
    lap = F.conv2d(ch, k, padding=1)
    return torch.abs(lap) * 5.0


def _xun(ch):
    """巽 — 方向/穿透度。最强方向超出其他方向的程度。"""
    device = ch.device

    k0 = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=ch.dtype, device=device)
    g0 = F.conv2d(ch, k0, padding=1)     # 0°
    g90 = F.conv2d(ch, k0.transpose(2,3), padding=1)  # 90°
    k45 = torch.tensor([[[[-2,-1,0],[-1,0,1],[0,1,2]]]], dtype=ch.dtype, device=device)
    g45 = F.conv2d(ch, k45, padding=1)
    k135 = torch.tensor([[[[0,1,2],[-1,0,1],[-2,-1,0]]]], dtype=ch.dtype, device=device)
    g135 = F.conv2d(ch, k135, padding=1)

    g = torch.stack([g0.abs().squeeze(1), g45.abs().squeeze(1),
                     g90.abs().squeeze(1), g135.abs().squeeze(1)], dim=1)  # [B,4,H,W]
    mx, _ = g.max(dim=1, keepdim=True)  # [B,1,H,W]
    # 减去其他三个方向的均值 = 纯方向性
    directionality = (mx - (g.sum(dim=1, keepdim=True) - mx) / 3.0).clamp(min=0)
    return directionality * 3.0


def _kan(ch):
    """坎 — 水/陷。多尺度低洼：中心比周围暗=凹陷。"""
    darkness = (1.0 - ch).clamp(min=0)
    _, _, H, W = ch.shape
    depression = torch.zeros_like(ch)
    for k in [7, 15, 31]:
        c = _box_filter(ch, k=k)
        s = _box_filter(ch, k=k*2+1)  # 奇数防尺寸漂移
        depression = depression + (s - c).clamp(min=0)
    return depression * darkness * 5.0


def _li(ch):
    """离 — 光明/炽热度。颜色投影已挑出暖色，亮度即离强度。"""
    return ch


def _gen(ch):
    """艮 — 山/阻隔/块度。大范围纹理特性突变的边界。"""
    lm = _box_filter(ch, k=15)
    texture = _box_filter((ch - lm)**2, k=15)
    # 平滑纹理图 → 只有大型纹理区域切换才被检测
    texture = _box_filter(texture, k=15)

    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=ch.dtype, device=ch.device)
    gx = F.conv2d(texture, sobel, padding=1)
    gy = F.conv2d(texture, sobel.transpose(2, 3), padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6) * 20.0


def _dui(ch):
    """兑 — 中心-环绕颜色对比度，无压缩"""
    device = ch.device
def _dui(ch):
    """兑 — 泽/开口/缺损。局部表面不连续 = 中心与周围差异。"""
    center = _box_filter(ch, k=5)
    surround = _box_filter(ch, k=21)
    return torch.abs(surround - center) * 5.0


# 算子注册表
BASE_OPS = {
    "qian": _qian, "kun": _kun, "zhen": _zhen, "xun": _xun,
    "kan": _kan, "li": _li, "gen": _gen, "dui": _dui,
}

BAGUA_NAMES = {
    "qian": "乾天", "kun": "坤地", "zhen": "震雷",
    "xun": "巽风", "kan": "坎水", "li": "离火",
    "gen": "艮山", "dui": "兑泽",
}

BAGUA_OPERATORS = {name: (BAGUA_NAMES[name], name) for name in BASE_OPS}


# ═══════════════════════════════════════════════════════════
# 固 定 颜 色 算 子
# ═══════════════════════════════════════════════════════════

class ColorFixedOperator(nn.Module):
    """
    固定颜色感应 + 固定几何算子 = 完整卦象检测器

    color_weights: [R, G, B] 固定系数，不参与训练
    算子从诞生起就知道自己该响应什么颜色的形态。
    """
    def __init__(self, base_fn, color_weights):
        super().__init__()
        self.base_fn = base_fn
        # register_buffer = 保存为模型参数的一部分，但不参与梯度更新
        self.register_buffer('weight',
            torch.tensor(color_weights, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        # 像素在卦象颜色方向上的投影 = 该象的程度值
        # clamp 保证程度值在 [0,1]，负数表示"完全不象"
        x_mod = (x * self.weight).sum(dim=1, keepdim=True).clamp(min=0)
        return self.base_fn(x_mod)


class ColorFixedOperatorLayer(nn.Module):
    """
    8 个固定颜色算子组成的算子层。
    每个算子的颜色偏好由八卦传统决定，一生不变。
    """
    def __init__(self):
        super().__init__()
        self.ops = nn.ModuleDict({
            name: ColorFixedOperator(fn, FIXED_BAGUA_COLORS[name])
            for name, fn in BASE_OPS.items()
        })

    def forward(self, x):
        return {name: op(x) for name, op in self.ops.items()}


# 兼容 pipeline.py 的导入
BaguaOperatorLayer = ColorFixedOperatorLayer

if __name__ == "__main__":
    import cv2 as cv
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    cv.circle(img, (112, 112), 70, (60, 60, 60), -1)
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    layer = ColorFixedOperatorLayer()
    r = layer(x)
    print("固定颜色算子测试：")
    for name in BASE_OPS:
        v = r[name][0, 0, 100:124, 100:124].mean().item()
        w = FIXED_BAGUA_COLORS[name]
        w_str = " ".join(f"{x:.2f}" for x in w)
        print(f"  {BAGUA_NAMES[name]:6s}: {v:.4f}  颜色=[{w_str}]")
