"""
八卦算子 — 颜色调制版 [VERSION=colormod]

每个算子前加一个可学习的 1×1 颜色卷积（3→1）。
算子本身保持灰度逻辑不变。
总新增参数：8×(3+1) = 32 个。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════
# 灰度算子（纯函数，与灰度版一致）
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


# 算子注册表（函数，用于ColorModulatedOperator）
BASE_OPS = {
    "qian": _qian,
    "kun": _kun,
    "zhen": _zhen,
    "xun": _xun,
    "kan": _kan,
    "li": _li,
    "gen": _gen,
    "dui": _dui,
}

BAGUA_NAMES = {
    "qian": "乾天", "kun": "坤地", "zhen": "震雷",
    "xun": "巽风", "kan": "坎水", "li": "离火",
    "gen": "艮山", "dui": "兑泽",
}

# 兼容 pipeline.py 的导入
BAGUA_OPERATORS = {name: (BAGUA_NAMES[name], name) for name in BASE_OPS}


# ═══════════════════════════════════════════════════════════
# 颜色调制算子包装
# ═══════════════════════════════════════════════════════════

class ColorModulatedOperator(nn.Module):
    """
    在灰度算子前加一个可学习的 1×1 颜色卷积（3→1）。

    初始权重决定每个算子偏好的颜色：
      [1,0,0] → 只看红色
      [0,1,0] → 只看绿色
      [0,0,1] → 只看蓝色
      [1/3,1/3,1/3] → 等权=灰度
    """
    def __init__(self, base_fn, name="", init_rgb=None):
        super().__init__()
        self.base_fn = base_fn
        self.name = name
        self.color_conv = nn.Conv2d(3, 1, 1, bias=True)

        if init_rgb is not None:
            with torch.no_grad():
                self.color_conv.weight.copy_(torch.tensor(init_rgb,
                    dtype=torch.float32).view(1, 3, 1, 1))
                self.color_conv.bias.zero_()
        else:
            nn.init.constant_(self.color_conv.weight, 1/3)
            nn.init.zeros_(self.color_conv.bias)

    def forward(self, x):
        # x: [B, 3, H, W] → 颜色调制 → [B, 1, H, W] → 灰度算子
        x_mod = self.color_conv(x)
        return self.base_fn(x_mod)


class ColorModulatedOperatorLayer(nn.Module):
    """
    8 个颜色调制算子组成的算子层。
    每个算子有独立的 1×1 颜色卷积核。
    """
    def __init__(self):
        super().__init__()
        # 为每个算子分配不同的初始颜色偏好
        rgb_inits = {
            "qian": [1, 0, 0],       # 乾→红
            "kun":  [0, 1, 0],       # 坤→绿
            "zhen": [0, 0, 1],       # 震→蓝
            "xun":  [1, 1, 0],       # 巽→红+绿（黄）
            "kan":  [0, 1, 1],       # 坎→绿+蓝（青）
            "li":   [1, 0, 1],       # 离→红+蓝（紫）
            "gen":  [1, 0.5, 0],     # 艮→橙
            "dui":  [0.5, 0, 1],     # 兑→紫蓝
        }

        self.ops = nn.ModuleDict({
            name: ColorModulatedOperator(fn, name=name, init_rgb=rgb_inits[name])
            for name, fn in BASE_OPS.items()
        })

    def forward(self, x):
        return {name: op(x) for name, op in self.ops.items()}


if __name__ == "__main__":
    import cv2 as cv
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    cv.circle(img, (112, 112), 70, (60, 60, 60), -1)
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    layer = ColorModulatedOperatorLayer()
    r = layer(x)
    print("颜色调制算子测试：")
    for name, (cn) in BAGUA_NAMES.items():
        v = r[name][0, 0, 100:124, 100:124].mean().item()
        w = layer.ops[name].color_conv.weight.data.view(-1).tolist()
        w_str = " ".join(f"{x:.2f}" for x in w)
        print(f"  {cn:6s}: {v:.4f}  权重=[{w_str}]")
