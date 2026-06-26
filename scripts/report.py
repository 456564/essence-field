"""
理解报告 — 展示模型对物质的理解

用法: python scripts/report.py photo.jpg
输出: test_output/xxx_report.png
    左: 原图 + 物质边界叠加
    中: 每种物质的物理解释（算子签名 → 物理含义 → 例子）
    右: 关键证据（算子热度图）
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 256


PALETTE = np.array([
    [230, 80, 80], [80, 180, 80], [80, 80, 220],
    [220, 180, 40], [180, 80, 200], [80, 200, 200],
]) / 255.0

OP_NAMES = ['qian','kun','zhen','xun','kan','li','gen','dui']


def run(img_path):
    pipe = BaguaPipeline(d=8).to(DEVICE).eval()
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255

    with torch.no_grad():
        field, edge_metric, norm_ops = essence_field_compute(
            x, pipe, presmooth_sigma=3.0, edge_scale=5.0, use_appearance=True)
    field_smooth, conv = edge_aware_diffusion(
        field, edge_metric, n_iters=20, alpha=0.15)
    labels, _, _, _ = field_to_materials(field, edge_metric, n_clusters=3)
    materials = describe_material(labels, field_smooth)

    # === 绘图 ===
    fig = plt.figure(figsize=(18, 10))

    # 左 1/3: 原图 + 物质分割对比
    ax1 = fig.add_axes([0.02, 0.05, 0.28, 0.90])
    # 上半: 原图
    ax1a = fig.add_axes([0.02, 0.52, 0.28, 0.43])
    ax1a.imshow(img_small); ax1a.set_title('Input Image'); ax1a.axis('off')

    # 下半: 多色分割
    ax1b = fig.add_axes([0.02, 0.05, 0.28, 0.43])
    seg = np.zeros((SIZE, SIZE, 3))
    for k in range(min(6, len(materials))):
        seg[labels == k] = PALETTE[k]
    bg = img_small.astype(float)/255 * 0.35
    ax1b.imshow(np.clip(bg + seg * 0.65, 0, 1))
    ax1b.set_title(f'Material Segmentation ({len(materials)} found)')
    ax1b.axis('off')

    # 右 2/3: 物理解释
    ax2 = fig.add_axes([0.33, 0.05, 0.65, 0.90])
    ax2.axis('off')

    y = 0.95
    ax2.text(0.02, y,
             f'ESSENCE FIELD — Material Understanding Report',
             fontsize=16, fontweight='bold', transform=ax2.transAxes)
    y -= 0.06
    ax2.text(0.02, y, f'{os.path.basename(img_path)} ({w}x{h})  |  '
             f'Diffusion converged: {conv[-1]:.4f}  |  {len(materials)} materials found',
             fontsize=9, color='gray', transform=ax2.transAxes)

    y -= 0.08
    for i, m in enumerate(materials):
        color = PALETTE[i % 6]
        # 物质标题
        ax2.text(0.02, y, f'Material {i+1}  ({m["area"]*100:.0f}% of image)',
                 fontsize=13, fontweight='bold', color=color, transform=ax2.transAxes)
        y -= 0.04

        # 算子签名 + 物理解释
        # m['prototype'] is 14-dim if use_appearance=True, 8-dim if not
        proto = m.get('prototype', np.zeros(8))
        C = len(proto)
        if C >= 14:
            # 分离表象和抽象
            app_proto = proto[:6]
            geo_proto = proto[6:]
            app_names = ['亮对比','色相对比','纹理粗细','边缘密度','方向一致','高光']
            geo_names = OP_NAMES

            # 表象层
            app_top = np.argsort(-app_proto)[:2]
            app_str = ', '.join(f'{app_names[j]}({app_proto[j]:.2f})' for j in app_top)
            ax2.text(0.04, y, f'Appearance: {app_str}',
                     fontsize=9, color='#555555', transform=ax2.transAxes)
            y -= 0.03

            # 抽象层
            geo_top = np.argsort(-geo_proto)[:3]
            for j in geo_top:
                name = geo_names[j]
                ax2.text(0.04, y, f'{name} = {geo_proto[j]:.3f}',
                         fontsize=10, fontweight='bold', transform=ax2.transAxes)
                y -= 0.03
            y -= 0.02
        else:
            geo_proto = proto
            geo_top = np.argsort(-geo_proto)[:3]
            for j in geo_top:
                name = OP_NAMES[j]
                ax2.text(0.04, y, f'{name} = {geo_proto[j]:.3f}',
                         fontsize=10, fontweight='bold', transform=ax2.transAxes)
                y -= 0.03
            y -= 0.02

    # 底部: 算子热度证据
    evidence_ax = fig.add_axes([0.02, 0.00, 0.96, 0.04])
    evidence_ax.axis('off')
    evidence_ax.text(0.5, 0.5,
                     '8 fixed geometric operators. Zero learned parameters. '
                     'Zero training. Field self-organization + clustering.',
                     ha='center', fontsize=8, color='#999999', style='italic',
                     transform=evidence_ax.transAxes)

    name = os.path.splitext(os.path.basename(img_path))[0]
    out = f'test_output/{name}_report.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Report: {out}')
    for i, m in enumerate(materials):
        print(f'  M{i+1}: {m["area"]*100:.0f}%  [{m["top_operators"]}]')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    run(args.image)
