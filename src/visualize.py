"""
卦象声纳 — 64维场可视化调试面板

三个组件：
  1. 活跃度矩阵 — 8×8 格栅，每格 = 一个子网络的全局强度
  2. 主控强度地图 — 按卦选择，展示该卦在原图上的空间响应
  3. 训练演化视图 — 周期性记录，绘制子网络活动随时间变化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 64维 → 8算子映射（每个算子8个投影通道）
GUA_NAMES = ['乾qian', '坤kun', '震zhen', '巽xun',
             '坎kan', '离li', '艮gen', '兑dui']
GUA_COLORS = ['#E74C3C', '#2ECC71', '#3498DB', '#F1C40F',
              '#1ABC9C', '#E91E63', '#E67E22', '#9B59B6']
GUA_SYMBOLS = ['乾', '坤', '震', '巽', '坎', '离', '艮', '兑']


def _per_gua_intensity(field_64):
    """
    field_64: [B, 64, H, W] → 每个卦的全局强度 [8]
    64维中每8维对应一个算子
    """
    B, C, H, W = field_64.shape
    intensity = []
    for i in range(8):
        block = field_64[:, i*8:(i+1)*8, :, :]      # [B, 8, H, W]
        norm = block.norm(dim=1).mean()               # L2范数, 全局平均
        intensity.append(norm.item())
    return np.array(intensity)


def _per_gua_spatial(field_64, gua_idx):
    """
    field_64: [B, 64, H, W] → 指定卦的空间强度图 [H, W]
    """
    block = field_64[0, gua_idx*8:(gua_idx+1)*8, :, :]  # [8, H, W]
    return block.norm(dim=0).cpu().numpy()              # [H, W]


# ═══════════════════════════════════════════════════
# 组件1：活跃度矩阵 8×8
# ═══════════════════════════════════════════════════

def sonar_matrix(field_64, ax=None, title="卦象声纳"):
    """
    8×8 格栅：每行 = 一个卦，每列 = 该卦的一个投影通道
    颜色深浅 = 该子网络的全局平均范数
    """
    B, C, H, W = field_64.shape
    # 计算每个子网络的全局范数 → [8, 8]
    matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            ch = field_64[0, i*8 + j, :, :]
            matrix[i, j] = ch.norm().item()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_ylabel("算子（卦）")
    ax.set_xlabel("投影通道")

    # 卦名 + 符号
    row_max = matrix.mean(axis=1)
    for i in range(8):
        ax.text(-1.5, i, f"{GUA_SYMBOLS[i]} {GUA_NAMES[i]}", ha='right', va='center',
                fontsize=9, color='black',
                fontweight='bold' if row_max[i] > row_max.mean() else 'normal')

    # 数值
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha='center', va='center',
                    fontsize=7, color='white' if matrix[i,j] > matrix.mean() else 'black')

    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label='子网络范数')
    return matrix


# ═══════════════════════════════════════════════════
# 组件2：主控强度地图（单卦空间分布）
# ═══════════════════════════════════════════════════

def activation_map(field_64, gua_idx, ax=None, title=None, cmap='hot'):
    """
    指定卦的空间强度热力图
    """
    spat = _per_gua_spatial(field_64, gua_idx)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    vmin, vmax = np.percentile(spat, 5), np.percentile(spat, 98)
    ax.imshow(spat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title or f"{GUA_SYMBOLS[gua_idx]} {GUA_NAMES[gua_idx]}")
    ax.axis('off')


def eight_gua_maps(field_64, figsize=(16, 8)):
    """8 卦空间强度全景图"""
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        activation_map(field_64, i, ax=ax,
                       title=f"{GUA_SYMBOLS[i]} {GUA_NAMES[i]} ({GUA_NAMES[i][:2]})")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════
# 组件3：训练演化视图
# ═══════════════════════════════════════════════════

class TrainingLogger:
    """在训练循环中周期性记录子网络强度"""
    def __init__(self):
        self.steps = []
        self.per_gua = {name: [] for name in GUA_NAMES}

    def log(self, step, field_64):
        """field_64: [B, 64, H, W]"""
        intensity = _per_gua_intensity(field_64)
        self.steps.append(step)
        for i, name in enumerate(GUA_NAMES):
            self.per_gua[name].append(intensity[i])

    def plot(self, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        for i, (name, color) in enumerate(zip(GUA_NAMES, GUA_COLORS)):
            if len(self.steps) > 0:
                ax.plot(self.steps, self.per_gua[name], color=color, label=name, linewidth=1.5)
        ax.legend(ncol=4, fontsize=8, loc='upper right')
        ax.set_xlabel("训练步")
        ax.set_ylabel("子网络活跃度（全局范数）")
        ax.set_title("卦象子网络训练演化")
        ax.grid(True, alpha=0.3)
        return fig


# ═══════════════════════════════════════════════════
# 全景面板（三个组件合并）
# ═══════════════════════════════════════════════════

# ═══════════════════════════════════════════════════
# 组件4：单卦叠加视图 + 最强卦复合图
# ═══════════════════════════════════════════════════

def overlay_gua_on_image(field_64, original_img, gua_idx, alpha=0.5, cmap=None):
    """
    在原图上叠加指定卦的强度热力图
    gua_idx: 0..7
    alpha: 混合透明度 (0=纯原图, 1=纯热力)
    cmap: None → 用卦的专属颜色；否则 matplotlib colormap
    返回 RGB uint8 图像
    """
    spat = _per_gua_spatial(field_64, gua_idx)
    vmin, vmax = np.percentile(spat, 2), np.percentile(spat, 98)
    normed = np.clip((spat - vmin) / (vmax - vmin + 1e-8), 0, 1)

    if cmap is None:
        color = np.array(plt.cm.colors.hex2color(GUA_COLORS[gua_idx]))
        heatmap = normed[..., np.newaxis] * color[np.newaxis, np.newaxis, :]
    else:
        from matplotlib.cm import get_cmap
        heatmap = get_cmap(cmap)(normed)[:, :, :3]

    img_norm = original_img / 255.0 if original_img.max() > 1 else original_img.copy()
    blended = img_norm * (1 - alpha) + heatmap * alpha
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def argmax_gua_composite(field_64, original_img=None):
    """
    最强卦复合图：每个像素的颜色 = 响应最强的卦的专属颜色
    强度值决定亮度（弱=暗，强=亮）
    """
    B, C, H, W = field_64.shape
    # 每个卦的空间强度 [8, H, W]
    gua_strength_raw = torch.zeros(8, H, W, device=field_64.device)
    for i in range(8):
        block = field_64[0, i*8:(i+1)*8, :, :]
        gua_strength_raw[i] = block.norm(dim=0)

    # 每个算子内部归一化 → 公平 argmax
    gua_strength_norm = gua_strength_raw.clone()
    for i in range(8):
        s = gua_strength_norm[i].view(-1)
        lo = s.kthvalue(max(1, int(0.02 * len(s)))).values
        hi = s.kthvalue(min(len(s), int(0.98 * len(s)))).values
        spread = hi - lo
        if spread > 1e-6:
            gua_strength_norm[i] = (gua_strength_norm[i] - lo) / spread

    max_idx = gua_strength_norm.argmax(dim=0)
    max_idx = max_idx.detach().cpu().numpy()
    # 强度用原始值调制亮度（边缘→亮、内部扩散→暗但可见）
    max_strength = gua_strength_raw.max(dim=0).values.detach().cpu().numpy()

    # 归一化强度
    s_min, s_max = np.percentile(max_strength, 2), np.percentile(max_strength, 98)
    normed = np.clip((max_strength - s_min) / (s_max - s_min + 1e-8), 0, 1) ** 1.2

    # 按卦着色
    colors = np.array([plt.cm.colors.hex2color(c) for c in GUA_COLORS])  # [8,3]
    rgb = colors[max_idx] * normed[..., np.newaxis]

    # 叠加原图灰度背景
    if original_img is not None:
        bg = original_img / 255.0 if original_img.max() > 1 else original_img.copy()
        bg_gray = bg.mean(axis=2, keepdims=True)
        rgb = rgb * 0.7 + bg_gray * 0.3

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def full_sonar_panel(field_64, original_img=None):
    """
    完整声纳面板：
      上排：原图 + 8×8 活跃度矩阵
      下排：8卦空间强度地图（4×2 网格）
    """
    fig = plt.figure(figsize=(18, 10))

    # 上排：原图 + 矩阵
    ax_img = fig.add_axes([0.03, 0.55, 0.20, 0.40])
    if original_img is not None:
        ax_img.imshow(original_img)
    ax_img.set_title("原图")
    ax_img.axis('off')

    ax_mat = fig.add_axes([0.28, 0.55, 0.42, 0.40])
    sonar_matrix(field_64, ax=ax_mat)

    # 下排：8卦空间图（4×2）
    for i in range(8):
        row = i // 4
        col = i % 4
        left = 0.04 + col * 0.24
        bottom = 0.05 + (1 - row) * 0.25
        ax = fig.add_axes([left, bottom, 0.22, 0.22])
        activation_map(field_64, i, ax=ax)

    return fig


# ═══════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.pipeline import BaguaPipeline
    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BaguaPipeline().to(device).eval()
    ckpt = torch.load("checkpoints_fixedcolor/bootstrap_epoch15.pth", map_location=device)
    model.fusion.A.data = ckpt['A']
    model.operator_layer.projections.load_state_dict(ckpt['proj'])

    img = cv2.imread("test_maccup.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (128, 128))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        field = model(x)

    full_sonar_panel(field, original_img=img_small)
    plt.savefig("test_output/sonar_panel.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ test_output/sonar_panel.png")
