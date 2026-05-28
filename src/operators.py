"""
八卦算子 — 灰度版
所有算子内部转灰度后分析，丢弃颜色信息。
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
    ksize = k
    pad = ksize // 2
    weight = torch.ones(1, 1, ksize, ksize, device=x.device) / (ksize*ksize)
    return F.conv2d(x, weight, padding=pad)


def qian_heaven(x):
    B, C, H, W = x.shape
    gray = x.mean(dim=1, keepdim=True)
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    gx = F.conv2d(gray, sobel, padding=1)
    gy = F.conv2d(gray, sobel.transpose(2,3), padding=1)
    energy = torch.sqrt(gx**2 + gy**2 + 1e-6)
    mask = torch.sigmoid((energy - 0.03) * 200)
    outputs = []
    for r in [2, 4, 6]:
        angles = torch.linspace(0, 2*np.pi, 12, device=device)
        off_y = (r * torch.sin(angles)).round().long()
        off_x = (r * torch.cos(angles)).round().long()
        pad = r + 2
        padded = F.pad(gray, [pad]*4, mode='reflect')
        samples = []
        for dy, dx in zip(off_y, off_x):
            shifted = torch.roll(padded, (dy.item(), dx.item()), dims=(2,3))
            shifted = shifted[:, :, pad:pad+H, pad:pad+W]
            samples.append(shifted)
        samples = torch.stack(samples, dim=1)
        outputs.append(torch.exp(-samples.var(dim=1, unbiased=False) * 10))
    return torch.stack(outputs, dim=0).mean(dim=0) * mask


def kun_earth(x):
    gray = x.mean(dim=1, keepdim=True)
    local_mean = _box_filter(gray, k=9)
    local_var = _box_filter((gray - local_mean)**2, k=9)
    global_var = gray.var(dim=[2,3], keepdim=True) + 1e-6
    return torch.exp(-local_var / global_var)


def zhen_thunder(x):
    gray = x.mean(dim=1, keepdim=True)
    kernel = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]],
                          dtype=x.dtype, device=x.device)
    lap = F.conv2d(gray, kernel, padding=1)
    return torch.tanh(torch.abs(lap) * 5)


def xun_wind(x):
    gray = x.mean(dim=1, keepdim=True)
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=device)
    kernels = [
        torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=x.dtype, device=device),
        torch.tensor([[[[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]], dtype=x.dtype, device=device),
        torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=x.dtype, device=device),
        torch.tensor([[[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]]], dtype=x.dtype, device=device),
    ]
    grads = [F.conv2d(gray, k, padding=1) for k in kernels]
    grads = torch.stack(grads, dim=1)
    grad_mean = grads.mean(dim=1, keepdim=True)
    grad_var = ((grads - grad_mean)**2).mean(dim=1)
    thinness = torch.exp(-grad_var * 5)
    gx_all = F.conv2d(gray, sobel, padding=1)
    gy_all = F.conv2d(gray, sobel.transpose(2,3), padding=1)
    energy = torch.sigmoid((torch.sqrt(gx_all**2 + gy_all**2 + 1e-6) - 0.03) * 200)
    return thinness * energy


def kan_water(x):
    gray = x.mean(dim=1, keepdim=True)
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    gx = F.conv2d(gray, sobel, padding=1)
    gy = F.conv2d(gray, sobel.transpose(2,3), padding=1)
    gm = torch.sqrt(gx**2 + gy**2 + 1e-6)
    nx = gx / (gm + 1e-6)
    ny = gy / (gm + 1e-6)
    nxx = F.conv2d(nx, sobel, padding=1)
    nyy = F.conv2d(ny, sobel.transpose(2,3), padding=1)
    return torch.tanh(torch.abs(nxx + nyy) * 10)


def li_fire(x):
    gray = x.mean(dim=1, keepdim=True)
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=x.device)
    gx = F.conv2d(gray, sobel, padding=1)
    gy = F.conv2d(gray, sobel.transpose(2,3), padding=1)
    grad_energy = torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 5)
    color_var = x.std(dim=1, keepdim=True)
    saturation = torch.tanh(color_var * 5)
    brightness = torch.sigmoid((gray - 0.3) * 5)
    return grad_energy * 0.4 + saturation * 0.3 + brightness * 0.3


def gen_mountain(x):
    gray = x.mean(dim=1, keepdim=True)
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=device)
    ps = 15
    pad = ps // 2
    padded = F.pad(gray, [pad]*4, mode='reflect')
    patches = F.unfold(padded, kernel_size=ps, stride=1)
    patch_var = patches.var(dim=1, unbiased=False).view(-1, 1, gray.shape[2], gray.shape[3])
    blockiness = torch.tanh(patch_var * 20)
    grad_x = F.conv2d(patch_var, sobel, padding=1)
    grad_y = F.conv2d(patch_var, sobel.transpose(2,3), padding=1)
    boundary = torch.tanh(torch.sqrt(grad_x**2 + grad_y**2 + 1e-6) * 10)
    return torch.max(blockiness * 0.5, boundary * 0.8)


def dui_lake(x):
    gray = x.mean(dim=1, keepdim=True)
    device = x.device
    sobel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                         dtype=x.dtype, device=device)
    gx_all = F.conv2d(gray, sobel, padding=1)
    gy_all = F.conv2d(gray, sobel.transpose(2,3), padding=1)
    energy_mask = torch.sigmoid((torch.sqrt(gx_all**2 + gy_all**2 + 1e-6) - 0.03) * 200)
    center = _box_filter(gray, k=5)
    surround = _box_filter(gray, k=15)
    contrast = surround - center
    concavity = torch.sigmoid((contrast - 0.02) * 40) * energy_mask
    gx = F.conv2d(contrast, sobel, padding=1)
    gy = F.conv2d(contrast, sobel.transpose(2,3), padding=1)
    asymmetry = torch.tanh(torch.sqrt(gx**2 + gy**2 + 1e-6) * 10) * energy_mask
    return torch.max(concavity, asymmetry)


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
