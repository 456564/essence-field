"""
多色物质分割可视化 —— 一图看清所有物质边界

每张图输出：
  Row 1: 原图 | 多色物质分割 | 边缘度规 | 场范数(扩散后)
  Row 2: 前4个算子热度图 (归一化响应)
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
                             field_to_materials, describe_material)
from src.operators import BASE_OPS, BAGUA_NAMES

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 160
DATA = 'data/caltech101/101_ObjectCategories'
OP_NAMES = list(BASE_OPS.keys())


def process_and_show(img_path, pipe, save_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255

    with torch.no_grad():
        field_8, edge_metric, norm_ops = essence_field_compute(
            x, pipe, presmooth_sigma=3.0, edge_scale=5.0)
    field_smooth, conv = edge_aware_diffusion(
        field_8, edge_metric, n_iters=20, alpha=0.15)
    labels, _, _, _ = field_to_materials(
        field_8, edge_metric, n_clusters=3)
    materials = describe_material(labels, field_smooth)

    # 算子热度（归一化后）
    op_heatmaps = {}
    for name in OP_NAMES:
        v = norm_ops[name][0,0].cpu().numpy()
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        op_heatmaps[name] = v

    edge_np = edge_metric[0,0].cpu().numpy()
    norms_np = field_smooth[0].norm(dim=0).cpu().numpy()
    # 自适应选精度
    n_show = min(4, len(materials))

    # === 画图 ===
    fig = plt.figure(figsize=(14, 7))

    # Row 0: 原图 + 多色分割 + 边缘 + 范数
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(img_small); ax.set_title('Original', fontsize=11); ax.axis('off')

    ax = fig.add_subplot(2, 4, 2)
    # 多色物质分割：每个物质不同颜色
    seg = np.zeros((SIZE, SIZE, 3), dtype=float)
    # 8色方案（色盲友好）
    palette = np.array([
        [230, 80, 80],    # 红
        [80, 180, 80],    # 绿
        [80, 80, 220],    # 蓝
        [220, 180, 40],   # 金
        [180, 80, 200],   # 紫
        [80, 200, 200],   # 青
        [200, 120, 60],   # 橙
        [160, 160, 160],  # 灰
    ]) / 255.0
    for k in range(n_show):
        seg[labels == k] = palette[k % 8]
    # 原图淡底 + 分割颜色
    bg = img_small.astype(float) / 255 * 0.4
    seg_blend = np.clip(bg + seg * 0.6, 0, 1)
    ax.imshow(seg_blend)
    ax.set_title(f'{len(materials)} materials', fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(edge_np, cmap='inferno')
    ax.set_title('Edge metric', fontsize=11); ax.axis('off')

    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(norms_np, cmap='hot',
              vmin=np.percentile(norms_np, 5), vmax=np.percentile(norms_np, 95))
    ax.set_title('Field norm (diffused)', fontsize=11); ax.axis('off')

    # Row 1: 前 4 个算子的热度图
    top_ops = ['kun', 'qian', 'zhen', 'xun']
    for i, name in enumerate(top_ops):
        ax = fig.add_subplot(2, 4, 5 + i)
        ax.imshow(op_heatmaps[name], cmap='hot')
        ax.set_title(f'{name}', fontsize=11); ax.axis('off')

    # 物质图例
    for k in range(n_show):
        m = materials[k]
        y_pos = 0.95 - k * 0.22
        color = palette[k % 8]
        fig.text(0.72, y_pos,
                 f'M{k+1}: {m["area"]*100:.0f}%  [{m["top_operators"][:45]}]',
                 fontsize=7, fontfamily='Microsoft YaHei',
                 color=color, transform=fig.transFigure)

    name = os.path.basename(img_path)
    plt.suptitle(name, fontsize=12, y=0.98)
    plt.subplots_adjust(left=0.05, right=0.70, top=0.92, bottom=0.05,
                        wspace=0.25, hspace=0.3)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    return materials


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='图片路径')
    parser.add_argument('--output', default='test_output/material_view.png')
    args = parser.parse_args()

    pipe = BaguaPipeline(d=8).to(DEVICE).eval()
    materials = process_and_show(args.image, pipe, args.output)
    print(f'Saved: {args.output}')
    for i, m in enumerate(materials):
        print(f'  M{i+1}: area={m["area"]*100:.0f}%  [{m["top_operators"]}]')


if __name__ == '__main__':
    main()
