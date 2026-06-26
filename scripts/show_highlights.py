"""
物质雷达图 + 原型像素 — 最直观的证据展示

每个物质显示:
  1. 8轴雷达图 (算子强度)
  2. 该物质中最"典型"的像素样例

用法: python scripts/show_highlights.py photo.jpg
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch, cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from src.pipeline import BaguaPipeline
from src.field_core import (essence_field_compute, edge_aware_diffusion,
                             field_to_materials)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 256
OP_NAMES = ['qian','kun','zhen','xun','kan','li','gen','dui']


def make_radar(ax, values, color, title, max_val=None):
    """画单物质雷达图"""
    N = len(values)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    vals = values.tolist() + values[:1]

    ax.fill(angles, vals, alpha=0.25, color=color)
    ax.plot(angles, vals, color=color, linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(OP_NAMES, fontsize=7)
    if max_val is not None:
        ax.set_ylim(0, max_val * 1.1)
    ax.set_title(title, fontsize=11, fontweight='bold', color=color)


def find_example_pixels(field_smooth, labels, k, n_samples=9):
    """找物质k中最典型的像素（离簇中心最近的像素）"""
    C = field_smooth.shape[1]
    vec = field_smooth[0].permute(1,2,0).reshape(-1, C).cpu().numpy()
    mask = (labels.flatten() == k)
    if mask.sum() == 0:
        return None
    vec_k = vec[mask]
    center = vec_k.mean(axis=0)
    dists = np.linalg.norm(vec_k - center, axis=1)
    idx_in_k = np.argsort(dists)[:n_samples]
    flat_indices = np.where(mask)[0][idx_in_k]
    ys = flat_indices // labels.shape[1]
    xs = flat_indices % labels.shape[1]
    return list(zip(ys, xs))


def run(img_path, output_path=None):
    pipe = BaguaPipeline(d=8).to(DEVICE).eval()
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255

    with torch.no_grad():
        field, edge_metric, _ = essence_field_compute(
            x, pipe, presmooth_sigma=3.0, edge_scale=5.0, use_appearance=True)
    field_smooth, conv = edge_aware_diffusion(
        field, edge_metric, n_iters=20, alpha=0.15)
    labels, _, _, prototypes = field_to_materials(
        field, edge_metric, n_clusters=3)

    C = field_smooth.shape[1]
    K = len(prototypes)
    geo_protos = prototypes[:, 6:] if C >= 14 else prototypes

    palette = np.array([
        [230, 80, 80], [80, 180, 80], [80, 80, 220],
        [220, 180, 40], [180, 80, 200], [80, 200, 200],
    ]) / 255.0

    # === 1×3 布局：分割图 | 雷达图对比 | 典型像素 ===
    fig = plt.figure(figsize=(18, 6))

    # 左: 分割叠加图
    ax_seg = fig.add_axes([0.02, 0.05, 0.28, 0.90])
    seg = np.zeros((SIZE, SIZE, 3))
    for k in range(K):
        seg[labels == k] = palette[k % 6]
    bg = img_small.astype(float)/255 * 0.3
    ax_seg.imshow(np.clip(bg + seg * 0.7, 0, 1))
    ax_seg.set_title('Material Segmentation', fontsize=12)
    ax_seg.axis('off')

    # 中: 雷达图对比（所有物质画在同一雷达图上对比）
    ax_radar = fig.add_axes([0.33, 0.15, 0.33, 0.75], projection='polar')
    max_val = max(p.max() for p in geo_protos) * 1.15
    N = len(OP_NAMES)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(OP_NAMES, fontsize=9)
    ax_radar.set_ylim(0, max_val)

    for k in range(K):
        vals = geo_protos[k].tolist() + geo_protos[k][:1].tolist()
        ax_radar.fill(angles, vals, alpha=0.12, color=palette[k])
        ax_radar.plot(angles, vals, color=palette[k], linewidth=2.5,
                      label=f'M{k+1}')
    ax_radar.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.15, 1.1))
    ax_radar.set_title('Operator Radar (geometric signature)', fontsize=12, pad=15)

    # 右: 典型像素样例
    for k in range(K):
        ax_px = fig.add_axes([0.69 + (k % 2) * 0.155, 0.45 - (k // 2) * 0.40,
                              0.14, 0.35])
        examples = find_example_pixels(field_smooth, labels, k, n_samples=9)
        if examples is None:
            ax_px.axis('off')
            continue

        # 3×3 网格显示典型像素的邻域
        for i, (py, px) in enumerate(examples[:9]):
            r, c = divmod(i, 3)
            # 取 20×20 邻域
            y0 = max(0, py - 10); y1 = min(SIZE, py + 10)
            x0 = max(0, px - 10); x1 = min(SIZE, px + 10)
            patch = img_small[y0:y1, x0:x1]

            # 在 3×3 子图中显示（手动放置）
            ps = 0.09
            ax_patch = fig.add_axes([0.69 + (k % 2) * 0.155 + c * ps,
                                     0.45 - (k // 2) * 0.40 + (2 - r) * ps,
                                     ps * 0.9, ps * 0.9])
            ax_patch.imshow(patch)
            ax_patch.axis('off')

        # 标颜色框
        for spine in ax_px.spines.values():
            spine.set_color(palette[k])
            spine.set_linewidth(3)
        ax_px.set_xticks([]); ax_px.set_yticks([])

    name = os.path.splitext(os.path.basename(img_path))[0]
    out = output_path or f'test_output/{name}_radar.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
    for k in range(K):
        p = geo_protos[k]
        top3 = np.argsort(-p)[:3]
        top_str = ', '.join(f'{OP_NAMES[j]}={p[j]:.3f}' for j in top3)
        print(f'  M{k+1}: {top_str}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    run(args.image, args.output)
