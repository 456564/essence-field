"""
物理算子声纳 — 8物理算子可视化调试面板

组件:
  1. 活跃度矩阵 — 8×8 格栅
  2. 强度地图 — 每个算子的空间响应
  3. 原图+最强算子叠加
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from .operators import dong, gang, cu, rou, ju, dist, yang, yin, void_prob

OP_NAMES = ['dong', 'gang', 'cu', 'rou', 'ju', 'dist', 'yang', 'yin']
OP_LABELS = ['动(梯度)', '刚(边界)', '粗(纹理)', '柔(渐变)',
             '聚(围合)', '距(边距)', '阳(实体)', '阴(虚空)']
OP_COLORS = ['#E74C3C', '#F39C12', '#3498DB', '#1ABC9C',
             '#2ECC71', '#9B59B6', '#E91E63', '#34495E']


def _per_op_spatial(base_9ch, op_idx):
    """base_9ch: [B, 9, H, W] from PhysicalOperatorLayer → 指定算子的空间强度 [H,W]"""
    return base_9ch[0, op_idx].cpu().numpy()


# ═══════════════════════════════════════════════════
# 组件1：算子响应全景图（8个算子并列显示）
# ═══════════════════════════════════════════════════

def operator_panel(base_9ch, figsize=(18, 6)):
    """
    8算子空间响应全景图 + void_prob
    base_9ch: [B, 9, H, W] from PhysicalOperatorLayer
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    for i in range(8):
        r, c = divmod(i, 5)
        spat = _per_op_spatial(base_9ch, i)
        ax = axes[r, c]
        vmin, vmax = np.percentile(spat, 2), np.percentile(spat, 98)
        ax.imshow(spat, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(OP_LABELS[i], fontsize=9)
        ax.axis('off')
    # 第9通道 = void_prob
    vp = _per_op_spatial(base_9ch, 8)
    axes[1, 4].imshow(vp, cmap='plasma', vmin=0, vmax=1)
    axes[1, 4].set_title("void_prob(虚空)", fontsize=9)
    axes[1, 4].axis('off')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════
# 组件2：单算子-原图叠加
# ═══════════════════════════════════════════════════

def overlay_op(base_9ch, op_idx, img_orig, alpha=0.5):
    """单个算子热力图叠加在原图上"""
    spat = _per_op_spatial(base_9ch, op_idx)
    img = img_orig / 255.0 if img_orig.max() > 1 else img_orig.copy()
    color = np.array(plt.cm.colors.hex2color(OP_COLORS[op_idx]))  # [3]
    vmin, vmax = np.percentile(spat, 2), np.percentile(spat, 98)
    normed = np.clip((spat - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heat = normed[..., np.newaxis] * color[np.newaxis, np.newaxis, :]
    blended = img * (1 - alpha) + heat * alpha
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def all_operator_overlays(base_9ch, img_orig):
    """8算子各自叠加在原图上，拼成一张图"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ov = overlay_op(base_9ch, i, img_orig)
        ax.imshow(ov)
        ax.set_title(OP_LABELS[i], fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════
# 组件3：原图 + 本质场范数
# ═══════════════════════════════════════════════════

def field_norm_map(field_66, img_orig, ax=None):
    """本质场66维L2范数热力图"""
    norms = field_66[0].norm(dim=0).cpu().numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    vmin, vmax = np.percentile(norms, 2), np.percentile(norms, 98)
    ax.imshow(norms, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title("本质场范数")
    ax.axis('off')
    return ax


def essence_panel(field_66, base_9ch, img_orig):
    """
    完整面板：原图 + 范数 + 8算子叠加 + void_prob
    """
    fig = plt.figure(figsize=(18, 12))
    # 原图
    ax = fig.add_axes([0.02, 0.7, 0.15, 0.25])
    img = img_orig / 255.0 if img_orig.max() > 1 else img_orig
    ax.imshow(img)
    ax.set_title("原图"); ax.axis('off')
    # 范数
    ax = fig.add_axes([0.20, 0.7, 0.15, 0.25])
    field_norm_map(field_66, img_orig, ax=ax)
    # 8算子叠加
    for i in range(8):
        r, c = i // 4, i % 4
        left, bottom = 0.02 + c * 0.24, 0.05 + (1 - r) * 0.30
        ax = fig.add_axes([left, bottom, 0.22, 0.27])
        ov = overlay_op(base_9ch, i, img_orig)
        ax.imshow(ov)
        ax.set_title(OP_LABELS[i], fontsize=8)
        ax.axis('off')
    # 第9通道 void_prob
    ax = fig.add_axes([0.74, 0.7, 0.22, 0.27])
    vp = _per_op_spatial(base_9ch, 8)
    ax.imshow(vp, cmap='plasma', vmin=0, vmax=1)
    ax.set_title("void_prob(虚空)", fontsize=9)
    ax.axis('off')
    return fig

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


def blended_gua_response(field_64):
    """PCA投影: 64维场→3维RGB, 同物相近=同色"""
    B, C, H, W = field_64.shape
    f = field_64[0].reshape(C, -1).t().cpu().numpy()  # [H*W, 64]

    # 采样加速: 取1/16像素做PCA
    idx = np.random.choice(f.shape[0], min(4096, f.shape[0]), replace=False)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3).fit(f[idx])
    rgb = pca.transform(f)  # [H*W, 3]

    # 每通道归一化到[0,1]
    for c in range(3):
        lo, hi = np.percentile(rgb[:,c], 2), np.percentile(rgb[:,c], 98)
        rgb[:,c] = np.clip((rgb[:,c] - lo) / (hi - lo + 1e-6), 0, 1)

    return (rgb.reshape(H, W, 3) * 255).astype(np.uint8)

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
    from src.pipeline import PhysicalPipeline
    from src.operators import PhysicalOperatorLayer
    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = PhysicalPipeline().to(device).eval()
    img = cv2.imread("test_maccup.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(cv2.resize(img_rgb, (224,224))).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    op_layer = PhysicalOperatorLayer()

    with torch.no_grad():
        base = op_layer(x)
        field = pipe(x)

    essence_panel(field, base, cv2.resize(img_rgb, (224,224)))
    plt.savefig("test_output/essence_panel.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ test_output/essence_panel.png")
