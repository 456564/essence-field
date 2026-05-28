"""
八卦算子 — RGB 版 [VERSION=rgb]
每个算子直接处理 RGB 三通道，各通道独立分析后取最大响应。
"""

import torch
import torch.nn.functional as F
import numpy as np


def _gaussian_kernel(size=7, sigma=1.5):
    ax = torch.arange(-size//2+1, size//2+1, dtype=torch.float32)
    g = torch.exp(-ax**2 / (2*sigma**2))
    g /= g.sum()
    return g.outer(g).view(1, 1, size, size)


def _box_filter(x, k=5):
    C = x.shape[1]
    weight = torch.ones(1, 1, k, k, device=x.device) / (k*k)
    weight = weight.expand(C, 1, k, k).contiguous()
    return F.conv2d(x, weight, padding=k//2, groups=C)


def qian_heaven(x):
    B, C, H, W = x.shape
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    outputs = []
    for c in range(C):
        ch = x[:, c:c+1, :, :]
        gx = F.conv2d(ch, sobel, padding=1)
        gy = F.conv2d(ch, sobel.transpose(2,3), padding=1)
        energy = torch.sqrt(gx**2 + gy**2 + 1e-6)
        mask = torch.sigmoid((energy - 0.03) * 200)
        ch_out = []
        for r in [2, 4, 6]:
            angles = torch.linspace(0, 2*np.pi, 12, device=device)
            off_y = (r * torch.sin(angles)).round().long()
            off_x = (r * torch.cos(angles)).round().long()
            pad = r + 2
            padded = F.pad(ch, [pad]*4, mode='reflect')
            samples = []
            for dy, dx in zip(off_y, off_x):
                shifted = torch.roll(padded, (dy.item(), dx.item()), dims=(2,3))
                shifted = shifted[:, :, pad:pad+H, pad:pad+W]
                samples.append(shifted)
            samples = torch.stack(samples, dim=1)
            ch_out.append(torch.exp(-samples.var(dim=1, unbiased=False) * 10))
        outputs.append(torch.stack(ch_out, dim=0).mean(dim=0) * mask)
    return torch.cat(outputs, dim=1).max(dim=1, keepdim=True)[0]


def kun_earth(x):
    local_mean = _box_filter(x, k=9)
    local_var = _box_filter((x - local_mean)**2, k=9)
    global_var = x.var(dim=[2,3], keepdim=True) + 1e-6
    flatness = torch.exp(-local_var / global_var)
    return flatness.max(dim=1, keepdim=True)[0]


def zhen_thunder(x):
    C = x.shape[1]
    kernel = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]],
                          dtype=x.dtype, device=x.device)
    kernel = kernel.expand(C, -1, -1, -1).contiguous()
    lap = F.conv2d(x, kernel, padding=1, groups=C)
    return lap.abs().max(dim=1, keepdim=True)[0].tanh_().mul_(5).tanh_()


def xun_wind(x):
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=device)
    kernel_defs = [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    ]
    kernels = [torch.tensor([[k]], dtype=x.dtype, device=device) for k in kernel_defs]
    outputs = []
    for c in range(x.shape[1]):
        ch = x[:, c:c+1, :, :]
        grads = [F.conv2d(ch, k, padding=1) for k in kernels]
        grads = torch.stack(grads, dim=1)
        grad_mean = grads.mean(dim=1, keepdim=True)
        grad_var = ((grads - grad_mean)**2).mean(dim=1)
        thinness = torch.exp(-grad_var * 5)
        gx_all = F.conv2d(ch, sobel, padding=1)
        gy_all = F.conv2d(ch, sobel.transpose(2,3), padding=1)
        energy = torch.sigmoid((torch.sqrt(gx_all**2 + gy_all**2 + 1e-6) - 0.03) * 200)
        outputs.append(thinness * energy)
    return torch.cat(outputs, dim=1).max(dim=1, keepdim=True)[0]


def kan_water(x):
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    outputs = []
    for c in range(x.shape[1]):
        ch = x[:, c:c+1, :, :]
        gx = F.conv2d(ch, sobel, padding=1)
        gy = F.conv2d(ch, sobel.transpose(2,3), padding=1)
        gm = torch.sqrt(gx**2 + gy**2 + 1e-6)
        nx = gx / (gm + 1e-6)
        ny = gy / (gm + 1e-6)
        nxx = F.conv2d(nx, sobel, padding=1)
        nyy = F.conv2d(ny, sobel.transpose(2,3), padding=1)
        outputs.append(torch.tanh(torch.abs(nxx + nyy) * 10))
    return torch.cat(outputs, dim=1).max(dim=1, keepdim=True)[0]


def li_fire(x):
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    C = x.shape[1]
    kernel_x = sobel.expand(C, -1, -1, -1).contiguous()
    kernel_y = sobel.transpose(2,3).expand(C, -1, -1, -1).contiguous()
    gx = F.conv2d(x, kernel_x, padding=1, groups=C)
    gy = F.conv2d(x, kernel_y, padding=1, groups=C)
    grad_energy = torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 5)
    grad_energy = grad_energy.max(dim=1, keepdim=True)[0]
    color_var = x.std(dim=1, keepdim=True)
    saturation = torch.tanh(color_var * 5)
    brightness = torch.sigmoid((x.mean(dim=1, keepdim=True) - 0.3) * 5)
    return grad_energy * 0.4 + saturation * 0.3 + brightness * 0.3


def gen_mountain(x):
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=device)
    outputs = []
    for c in range(x.shape[1]):
        ch = x[:, c:c+1, :, :]
        _, _, Hc, Wc = ch.shape
        ps = 15
        pad = ps // 2
        padded = F.pad(ch, [pad]*4, mode='reflect')
        patches = F.unfold(padded, kernel_size=ps, stride=1)
        patch_var = patches.var(dim=1, unbiased=False).view(-1, 1, Hc, Wc)
        blockiness = torch.tanh(patch_var * 20)
        grad_x = F.conv2d(patch_var, sobel, padding=1)
        grad_y = F.conv2d(patch_var, sobel.transpose(2,3), padding=1)
        boundary = torch.tanh(torch.sqrt(grad_x**2 + grad_y**2 + 1e-6) * 10)
        outputs.append(torch.max(blockiness * 0.5, boundary * 0.8))
    return torch.cat(outputs, dim=1).max(dim=1, keepdim=True)[0]


def dui_lake(x):
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=device)
    outputs = []
    for c in range(x.shape[1]):
        ch = x[:, c:c+1, :, :]
        gx_all = F.conv2d(ch, sobel, padding=1)
        gy_all = F.conv2d(ch, sobel.transpose(2,3), padding=1)
        energy_mask = torch.sigmoid((torch.sqrt(gx_all**2 + gy_all**2 + 1e-6) - 0.03) * 200)
        center = _box_filter(ch, k=5)
        surround = _box_filter(ch, k=15)
        contrast = surround - center
        concavity = torch.sigmoid((contrast - 0.02) * 40) * energy_mask
        gx = F.conv2d(contrast, sobel, padding=1)
        gy = F.conv2d(contrast, sobel.transpose(2,3), padding=1)
        asymmetry = torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 10) * energy_mask
        outputs.append(torch.max(concavity, asymmetry))
    return torch.cat(outputs, dim=1).max(dim=1, keepdim=True)[0]


BAGUA_OPERATORS = {
    "qian": ("乾天", qian_heaven),
    "kun": ("坤地", kun_earth),
    "zhen": ("震雷", zhen_thunder),
    "xun": ("巽风", xun_wind),
    "kan": ("坎水", kan_water),
    "li": ("离火", li_fire),
    "gen": ("艮山", gen_mountain),
    "dui": ("兑泽", dui_lake),
}


class BaguaOperatorLayer:
    def __call__(self, x):
        results = {}
        for name, (cn, fn) in BAGUA_OPERATORS.items():
            results[name] = fn(x)
        return results


if __name__ == "__main__":
    import cv2 as cv
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    cv.circle(img, (112, 112), 70, (60, 60, 60), -1)
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    op = BaguaOperatorLayer()
    results = op(x)
    for name, (cn, _) in BAGUA_OPERATORS.items():
        v = results[name][0, 0, 100:124, 100:124].mean().item()
        print(f"  {cn:6s}: {v:.4f}")
