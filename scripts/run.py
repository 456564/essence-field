"""
一键物质发现 — 拖入任意照片，输出多色物质分割图

用法: python scripts/run.py photo.jpg
输出: test_output/photo_materials.png
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


def run(img_path, presmooth_sigma=3.0, edge_scale=5.0, n_iters=20):
    pipe = BaguaPipeline(d=8).to(DEVICE).eval()

    img = cv2.imread(img_path)
    if img is None:
        print(f'Cannot read: {img_path}')
        return
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255

    # 三层本质场管线 — 表象 + 抽象 → 联合边 → 稳定态
    with torch.no_grad():
        field, edge_metric, norm_ops = essence_field_compute(
            x, pipe, presmooth_sigma=presmooth_sigma, edge_scale=edge_scale,
            use_appearance=True, use_global_context=False)
    field_smooth, conv = edge_aware_diffusion(
        field, edge_metric, n_iters=n_iters, alpha=0.15)
    labels, _, _, prototypes, domain_stats = field_to_materials(
        field, edge_metric)

    n_materials = len(prototypes)
    areas = [ds['area_pct'] for ds in domain_stats]

    # 多色分割图
    palette = np.array([
        [230, 80, 80], [80, 180, 80], [80, 80, 220],
        [220, 180, 40], [180, 80, 200], [80, 200, 200],
    ]) / 255.0

    seg = np.zeros((SIZE, SIZE, 3))
    for k in range(min(6, n_materials)):
        seg[labels == k] = palette[k]

    bg = img_small.astype(float) / 255 * 0.35
    seg_blend = np.clip(bg + seg * 0.65, 0, 1)

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_small); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(seg_blend)
    axes[1].set_title(f'Stable States ({n_materials} materials)'); axes[1].axis('off')
    axes[2].axis('off')

    text = f'Resolution: {orig_w}x{orig_h}  |  Diffusion: {conv[-1]:.4f}\n\n'
    text += f'Domains found: {n_materials} (field stable states)\n'
    for k in range(n_materials):
        ds = domain_stats[k]
        text += (f'M{k+1}: {ds["area_pct"]:.0f}%  '
                 f'AbsC={ds["abstract_consistency"]:.2f}  '
                 f'AppC={ds["appearance_consistency"]:.2f}\n')
    axes[2].text(0.05, 0.95, text, transform=axes[2].transAxes,
                 fontsize=11, fontfamily='Microsoft YaHei', va='top')

    name = os.path.splitext(os.path.basename(img_path))[0]
    out = f'test_output/{name}_materials.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Image: {orig_w}x{orig_h}  |  Diffusion: {conv[-1]:.4f}')
    print(f'Saved: {out}')
    print(f'Domains: {n_materials}')
    for k in range(n_materials):
        ds = domain_stats[k]
        print(f'  M{k+1}: {ds["area_pct"]:.0f}%  '
              f'AbsC={ds["abstract_consistency"]:.2f}  '
              f'AppC={ds["appearance_consistency"]:.2f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='一键物质发现')
    parser.add_argument('image', help='图片路径')
    parser.add_argument('--size', type=int, default=256, help='处理分辨率')
    parser.add_argument('--sigma', type=float, default=3.0, help='预平滑强度(越大越融合同物体)')
    parser.add_argument('--edge', type=float, default=5.0, help='边缘阻断强度')
    parser.add_argument('--iters', type=int, default=20, help='扩散轮数')
    args = parser.parse_args()
    SIZE = args.size
    # 覆盖全局默认值
    import src.field_core as fc
    run(args.image, args.sigma, args.edge, args.iters)
