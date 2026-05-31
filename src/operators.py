"""
物理算子 — 8 算子（四对互补）

  dong(梯度) ─→ jing(1-dong)
  gang(脊线) ─→ rou(渗透)
  ju(围合)  ─→ san(1-ju)
  yang(实体) ─→ yin(1-yang)
  + cu(纹理) + dist(到边距离)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _box_filter(x, k=5):
    C = x.shape[1]
    w = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    p = k // 2
    x_pad = F.pad(x, (p, p, p, p), mode='replicate')
    return F.conv2d(x_pad, w.repeat(C, 1, 1, 1), padding=0, groups=C)


def _sobel_magnitude(x):
    device = x.device
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
    gx = F.conv2d(x_pad, kx.repeat(3, 1, 1, 1), padding=0, groups=3)
    gy = F.conv2d(x_pad, ky.repeat(3, 1, 1, 1), padding=0, groups=3)
    return torch.norm(torch.sqrt(gx ** 2 + gy ** 2), dim=1, keepdim=True)


_MAX_SOBEL = 5.0


# ── 动 ──
def dong(x):
    mag = _sobel_magnitude(x)
    mag = _box_filter(mag, k=3)
    return torch.clamp(mag / _MAX_SOBEL, 0.0, 1.0)


def _jing_from_dong(d):
    return 1.0 - d


# ── 刚 ──
def _gang_from_dong(d):
    ridges = []
    for k in [3, 7, 15]:
        ridges.append((d - _box_filter(d, k=k)).clamp(min=0))
    ridge = torch.stack(ridges, dim=0).max(dim=0)[0]
    # Gray supplement
    return ridge


def _gang_enhanced(x):
    d = dong(x)
    ridge_color = _gang_from_dong(d)
    gray = x.mean(dim=1, keepdim=True)
    gray_dong = torch.clamp(_box_filter(_sobel_magnitude(gray.repeat(1,3,1,1)), k=3) / _MAX_SOBEL, 0.0, 1.0)
    ridge_gray = _gang_from_dong(gray_dong)
    return torch.max(ridge_color, ridge_gray * 0.5)


def gang(x):
    return _gang_enhanced(x)


# ── 粗 ──
def _cu_from_rgb_dong(x, dong_map):
    local_mean = _box_filter(x, k=5)
    residual_sq = (x - local_mean) ** 2
    ew = (1.0 - dong_map).clamp(min=1e-6)
    variance = _box_filter(residual_sq * ew, k=7) / (_box_filter(ew, k=7) + 1e-8)
    return torch.clamp(variance.mean(dim=1, keepdim=True) / 0.04, 0.0, 1.0)


def cu(x):
    return _cu_from_rgb_dong(x, dong(x))


# ── 柔 ──
def _rou_from_dong_gang_cu(d, g, c):
    rou_edge = (d - g).clamp(min=0)
    rou_smooth = (1.0 - c).clamp(min=0)
    return d * rou_edge + (1.0 - d) * rou_smooth


def rou(x):
    d = dong(x); g = _gang_from_dong(d); c = _cu_from_rgb_dong(x, d)
    return _rou_from_dong_gang_cu(d, g, c)


# ── 聚 ──
def _ju_from_gang_dist(g, dist_map):
    """ju = base_coverage × interior_score (no density — gang=0 inside kills interior)"""
    _, _, H, W = g.shape
    thresh = 0.01
    best = torch.zeros_like(g)
    for R in [max(H,W)//8, max(H,W)//5, max(H,W)//3]:
        R = max(R, 15)
        if R > max(H, W)//2: continue
        g_pad = F.pad(g, (R, R, R, R), mode='replicate')
        ml = F.max_pool2d(g_pad, (1, R), stride=1)[:, :, R:R+H, R:R+W]
        gfx = torch.flip(g_pad, [3])
        mr = torch.flip(F.max_pool2d(gfx, (1, R), stride=1)[:, :, R:R+H, R:R+W], [3])
        mu = F.max_pool2d(g_pad, (R, 1), stride=1)[:, :, R:R+H, R:R+W]
        gfy = torch.flip(g_pad, [2])
        md = torch.flip(F.max_pool2d(gfy, (R, 1), stride=1)[:, :, R:R+H, R:R+W], [2])
        # Opposite-direction pair: need matching edges, not random fragments
        h_pair = ((ml > thresh).float() * (mr > thresh).float())
        v_pair = ((mu > thresh).float() * (md > thresh).float())
        covered = (h_pair + v_pair) / 2.0
        best = torch.max(best, covered)
    base_ju = best.clamp(0.0, 1.0)
    # interior_score: absolute distance threshold (not relative)
    interior_score = torch.sigmoid((dist_map - 0.08) * 30.0)  # 边界→0, 内部→1
    # texture penalty: gang fragments → not enclosure
    gang_density = _box_filter(g, k=7)
    texture_penalty = (1.0 - gang_density * 10.0).clamp(0.0, 1.0)
    return (base_ju * interior_score * texture_penalty).clamp(0.0, 1.0)


def ju(x):
    d = dong(x); g = _gang_from_dong(d); t = _dist_from_gang(g)
    return _ju_from_gang_dist(g, t)


def san(x):
    return 1.0 - ju(x)


# ── 距 ──
def _dist_from_gang(g):
    import cv2
    B, _, H, W = g.shape
    max_dist = np.sqrt(H**2 + W**2)
    dist_batch = []
    for b in range(B):
        edge = (g[b, 0] > 0.01).cpu().numpy().astype(np.uint8)
        dt = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5)
        dt = np.clip(dt / (max_dist * 0.3), 0.0, 1.0)
        dist_batch.append(torch.from_numpy(dt.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(g.device))
    return torch.cat(dist_batch, dim=0)


def dist(x):
    return _dist_from_gang(_gang_from_dong(dong(x)))


# ── 阳 ──
def _yang_from_dong_gang_cu_ju(d, g, c, ju_val):
    """
    yang = ju × (1-dong) × (1-gang) × (1-cu)

    ju: 围合门控 — 必须有围合才有实体（斑马 j=0 → y=0）
    (1-dong): 非边缘 — 实体在变化带内侧
    (1-gang): 非边界线 — gang 高 = 几何不连续
    (1-cu): 非粗糙纹理 — 草地/毛发 cu 高 → 不是光滑实体表面
    """
    yang_raw = ju_val * (1.0 - d) * (1.0 - g.clamp(0.0, 1.0)) * (1.0 - c)
    # Gaussian smooth
    sigma = 3.0; ksize = int(sigma * 3) | 1; half = (ksize - 1) // 2
    gauss_k = torch.exp(-torch.arange(-half, half+1, device=d.device, dtype=d.dtype)**2 / (2*sigma**2))
    gauss_k = gauss_k / gauss_k.sum()
    gk = (gauss_k.view(1, 1, 1, -1) * gauss_k.view(1, 1, -1, 1)).repeat(1, 1, 1, 1)
    p = ksize // 2
    y_pad = F.pad(yang_raw, (p,p,p,p), mode='replicate')
    return F.conv2d(y_pad, gk, padding=0).clamp(0.0, 1.0)


def yang(x):
    d = dong(x); g = _gang_from_dong(d); c = _cu_from_rgb_dong(x, d)
    ju_val = _ju_from_gang_dist(g, _dist_from_gang(g))
    return _yang_from_dong_gang_cu_ju(d, g, c, ju_val)


# ── 阴 ──
def yin(x):
    return 1.0 - yang(x)


# ── 虚空概率 ──
def _void_prob(ju_val, c, jing_val, yang_val, g):
    """
    void_prob = ju × (1-cu) × jing × (1-yang) × (gang<0.01)

    围合内部 + 无纹理 + 静止 + 非实体表面 + 非边界线 = 高概率虚空(空腔)
    动物毛发: cu 高 → (1-cu) 低 → void_prob 低 → 粒子不进入
    杯腔: cu 低, jing 高, yang 低 → void_prob 高 → 粒子可进入
    """
    return (ju_val * (1.0 - c) * jing_val * (1.0 - yang_val) * (g < 0.01).float()).clamp(0.0, 1.0)


def void_prob(x):
    d = dong(x); g = _gang_from_dong(d); c = _cu_from_rgb_dong(x, d)
    t = _dist_from_gang(g); ju_v = _ju_from_gang_dist(g, t)
    yg = _yang_from_dong_gang_cu_ju(d, g, c, ju_v)
    jing_v = _jing_from_dong(d)
    return _void_prob(ju_v, c, jing_v, yg, g)


# ── 注册表 ──
PHYSICAL_OPERATORS = {
    'dong': dong, 'gang': gang, 'cu': cu, 'rou': rou,
    'ju': ju, 'dist': dist, 'yang': yang, 'yin': yin,
}
N_OPS = 8


# ── 算子层 ──
class PhysicalOperatorLayer(nn.Module):
    def forward(self, x):
        d = dong(x)
        g = _gang_from_dong(d)
        c = _cu_from_rgb_dong(x, d)
        r = _rou_from_dong_gang_cu(d, g, c)
        t = _dist_from_gang(g)
        ju_v = _ju_from_gang_dist(g, t)
        yg = _yang_from_dong_gang_cu_ju(d, g, c, ju_v)
        yn = 1.0 - yg
        vp = _void_prob(ju_v, c, _jing_from_dong(d), yg, g)
        return torch.cat([d, g, c, r, ju_v, t, yg, yn, vp], dim=1)  # [B,9,H,W]
