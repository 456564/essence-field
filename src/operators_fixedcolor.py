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


# ═══════════════════════════════════════════════════════════
# 固 定 颜 色 权 重（八卦传统颜色映射）
# ═══════════════════════════════════════════════════════════

# 每个算子天生只知道"自己该看什么颜色"。
# 权重 = [R, G, B] 系数，正=敏感，负=抑制，0=忽略
FIXED_BAGUA_COLORS = {
    'qian': [1.0, 0.5, 0.0],   # 乾→天→赤/玄黄→红黄
    'kun':  [0.0, 0.5, 0.0],   # 坤→地→黄/黑→绿
    'zhen': [0.0, 1.0, 0.5],   # 震→雷→青绿→绿蓝
    'xun':  [1.0, 1.0, 1.0],   # 巽→风→白→全色（无偏）
    'kan':  [0.0, 0.2, 1.0],   # 坎→水→黑/深蓝→蓝
    'li':   [1.0, 0.0, 0.0],   # 离→火→赤→红
    'gen':  [0.8, 0.6, 0.2],   # 艮→山→黄/棕→橙黄
    'dui':  [0.5, 0.5, 0.8],   # 兑→泽→白/蓝→泛蓝白
}


# ═══════════════════════════════════════════════════════════
# 灰 度 算 子（纯函数，与灰度版一致）
# ═══════════════════════════════════════════════════════════

def _box_filter(x, k=5):
    w = torch.ones(1, 1, k, k, device=x.device) / (k*k)
    return F.conv2d(x, w, padding=k//2)


def _qian(ch):
    B, C, H, W = ch.shape
    device = ch.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=ch.dtype, device=ch.device)
    gx = F.conv2d(ch, sobel, padding=1)
    gy = F.conv2d(ch, sobel.transpose(2,3), padding=1)
    mask = torch.sigmoid((torch.sqrt(gx**2 + gy**2 + 1e-6) - 0.03) * 200)
    out = []
    for r in [2, 4, 6]:
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
        out.append(torch.exp(-smp.var(dim=1, unbiased=False) * 10))
    return torch.stack(out, dim=0).mean(dim=0) * mask


def _kun(ch):
    m = _box_filter(ch, k=9)
    v = _box_filter((ch - m)**2, k=9)
    gv = ch.var(dim=[2,3], keepdim=True) + 1e-6
    return torch.exp(-v / gv)


def _zhen(ch):
    k = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]],
                     dtype=ch.dtype, device=ch.device)
    lap = F.conv2d(ch, k, padding=1)
    return torch.tanh(torch.abs(lap) * 5)


def _xun(ch):
    device = ch.device
    kd = [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    ]
    k4 = [torch.tensor([[k]], dtype=ch.dtype, device=ch.device) for k in kd]
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=ch.dtype, device=ch.device)
    gs = torch.stack([F.conv2d(ch, k, padding=1) for k in k4], dim=1)
    gv = ((gs - gs.mean(dim=1, keepdim=True))**2).mean(dim=1)
    tn = torch.exp(-gv * 5)
    gx = F.conv2d(ch, sobel, padding=1)
    gy = F.conv2d(ch, sobel.transpose(2,3), padding=1)
    en = torch.sigmoid((torch.sqrt(gx**2 + gy**2 + 1e-6) - 0.03) * 200)
    return tn * en


def _kan(ch):
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=ch.dtype, device=ch.device)
    gx = F.conv2d(ch, sobel, padding=1)
    gy = F.conv2d(ch, sobel.transpose(2,3), padding=1)
    gm = torch.sqrt(gx**2 + gy**2 + 1e-6)
    nx = gx / (gm + 1e-6)
    ny = gy / (gm + 1e-6)
    nxx = F.conv2d(nx, sobel, padding=1)
    nyy = F.conv2d(ny, sobel.transpose(2,3), padding=1)
    return torch.tanh(torch.abs(nxx + nyy) * 10)


def _li(ch):
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=ch.dtype, device=ch.device)
    gx = F.conv2d(ch, sobel, padding=1)
    gy = F.conv2d(ch, sobel.transpose(2,3), padding=1)
    return torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 5)


def _gen(ch):
    device = ch.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=ch.dtype, device=ch.device)
    _, _, Hc, Wc = ch.shape
    ps, pd = 15, 7
    pad = F.pad(ch, [pd]*4, mode='reflect')
    pat = F.unfold(pad, kernel_size=ps, stride=1)
    pv = pat.var(dim=1, unbiased=False).view(-1, 1, Hc, Wc)
    bl = torch.tanh(pv * 20)
    gx = F.conv2d(pv, sobel, padding=1)
    gy = F.conv2d(pv, sobel.transpose(2,3), padding=1)
    bd = torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 10)
    return torch.max(bl * 0.5, bd * 0.8)


def _dui(ch):
    device = ch.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=ch.dtype, device=ch.device)
    gx = F.conv2d(ch, sobel, padding=1)
    gy = F.conv2d(ch, sobel.transpose(2,3), padding=1)
    em = torch.sigmoid((torch.sqrt(gx**2 + gy**2 + 1e-6) - 0.03) * 200)
    ctr = _box_filter(ch, k=5)
    sr = _box_filter(ch, k=15)
    ct = sr - ctr
    cv = torch.sigmoid((ct - 0.02) * 40) * em
    gx2 = F.conv2d(ct, sobel, padding=1)
    gy2 = F.conv2d(ct, sobel.transpose(2,3), padding=1)
    asy = torch.tanh(torch.sqrt(gx2**2 + gy2**2 + 1e-6) * 10) * em
    return torch.max(cv, asy)


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
        # 固定颜色混合（无参数，不训练）
        x_mod = (x * self.weight).sum(dim=1, keepdim=True)
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
