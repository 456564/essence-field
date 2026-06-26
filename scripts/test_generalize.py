"""
零参数泛化性测试

批量跑多张图片，验证：
  8算子 → 归一化 → 预平滑 → 边缘扩散 → 聚类 → 物质发现

输出每张图的物质数量、主算子特征、可视化面板。
"""

import os, sys, argparse
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


def process_image(img_path, pipe, device, args):
    """处理单张图片，返回结果"""
    img = cv2.imread(img_path)
    if img is None:
        return None, f"无法读取: {img_path}"

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (args.size, args.size))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device)/255.0

    # 零参数管线
    field_8, edge_metric, norm_ops = essence_field_compute(
        x, pipe, presmooth_sigma=args.sigma, edge_scale=args.edge_scale)

    # 边缘感知扩散
    field_smooth, conv = edge_aware_diffusion(
        field_8, edge_metric, n_iters=args.n_iters, alpha=args.alpha)

    # 聚类
    labels, _, _, prototypes = field_to_materials(
        field_8, edge_metric, n_clusters=(args.n_clusters or None))

    materials = describe_material(labels, field_smooth)

    result = {
        'img': img_small,
        'path': img_path,
        'size': (h, w),
        'field_8': field_8,
        'field_smooth': field_smooth,
        'edge_metric': edge_metric,
        'norm_ops': norm_ops,
        'labels': labels,
        'materials': materials,
        'convergence': conv,
    }
    return result, None


def make_panel(result, save_path):
    """为单张图生成可视化面板"""
    img = result['img']
    labels = result['labels']
    materials = result['materials']
    conv = result['convergence']
    edge = result['edge_metric'][0,0].cpu().numpy()
    norms = result['field_smooth'][0].norm(dim=0).cpu().numpy()

    K = len(materials)
    fig = plt.figure(figsize=(12, 3 + 3 * ((K+2)//3)))

    # Row 0: 原图 + 边缘度规 + 场范数 + 收敛
    ax = fig.add_subplot(3, 4, 1)
    ax.imshow(img); ax.set_title('Original', fontsize=10); ax.axis('off')

    ax = fig.add_subplot(3, 4, 2)
    ax.imshow(edge, cmap='hot'); ax.set_title('Edge metric', fontsize=10); ax.axis('off')

    ax = fig.add_subplot(3, 4, 3)
    ax.imshow(norms, cmap='hot',
              vmin=np.percentile(norms, 2), vmax=np.percentile(norms, 98))
    ax.set_title('Field norm', fontsize=10); ax.axis('off')

    ax = fig.add_subplot(3, 4, 4)
    if conv:
        ax.plot(conv, 'b-o', ms=2)
        ax.set_title(f'Convergence (d={conv[-1]:.4f})', fontsize=10)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No diffusion', ha='center', va='center')

    # Row 1: 前四个算子的响应
    op_names = ['qian','kun','zhen','xun']
    for i, name in enumerate(op_names):
        ax = fig.add_subplot(3, 4, 5 + i)
        v = result['norm_ops'][name][0,0].cpu().numpy()
        ax.imshow(v, cmap='hot', vmin=np.percentile(v,2), vmax=np.percentile(v,98))
        ax.set_title(name, fontsize=10); ax.axis('off')

    # Row 2: 物质掩码
    colors = [(220,60,60), (60,180,60), (60,60,220), (220,180,40)]
    for i, m in enumerate(materials[:4]):
        ax = fig.add_subplot(3, 4, 9 + i)
        overlay = img.astype(float)/255 * 0.35
        mask_rgb = np.zeros_like(img, dtype=float)
        for c in range(3):
            mask_rgb[:,:,c] = m['mask'] * colors[i][c] / 255
        overlay += mask_rgb * 0.65
        ax.imshow(np.clip(overlay, 0, 1))
        area_str = f'{m["area"]*100:.0f}%'
        top_str = m['top_operators'][:40]
        ax.set_title(f'M{i+1}: {area_str}\n{top_str}', fontsize=8)
        ax.axis('off')

    img_name = os.path.basename(result['path'])
    plt.suptitle(f'{img_name}  |  {K} materials', fontsize=12)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                        wspace=0.3, hspace=0.4)
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description='零参数泛化性测试')
    parser.add_argument('images', nargs='*', help='图片路径列表')
    parser.add_argument('--dir', default=None, help='扫描目录下所有图片')
    parser.add_argument('--size', type=int, default=128, help='resize')
    parser.add_argument('--sigma', type=float, default=3.0, help='预平滑 sigma')
    parser.add_argument('--edge-scale', type=float, default=5.0, help='边缘度规强度')
    parser.add_argument('--n-iters', type=int, default=30, help='扩散轮数')
    parser.add_argument('--alpha', type=float, default=0.15, help='扩散步长')
    parser.add_argument('--n-clusters', type=int, default=0, help='聚类数(0=auto)')
    parser.add_argument('--output-dir', default='test_output', help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集图片：目录 + 命令行参数
    image_paths = []
    if args.dir:
        import glob as globmod
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            image_paths.extend(globmod.glob(os.path.join(args.dir, ext)))
    image_paths.extend(args.images)
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print("No images found. Use --dir PATH or list image paths.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  Images: {len(image_paths)}")

    pipe = BaguaPipeline(d=8).to(device).eval()
    # 零可学习参数

    summary = []
    for img_path in image_paths:
        name = os.path.basename(img_path)
        print(f"\n--- {name} ---")

        result, err = process_image(img_path, pipe, device, args)
        if err:
            print(f"  [SKIP] {err}")
            continue

        # 打印物质信息
        for i, m in enumerate(result['materials']):
            print(f"  M{i+1}: {m['area']*100:.1f}%  [{m['top_operators'][:60]}]")

        # 可视化
        out_path = os.path.join(args.output_dir,
                                f"{os.path.splitext(name)[0]}_materials.png")
        make_panel(result, out_path)

        # 汇总
        summary.append({
            'image': name,
            'n_materials': len(result['materials']),
            'top_material': result['materials'][0]['area'] if result['materials'] else 0,
            'converged': result['convergence'][-1] < 0.01 if result['convergence'] else False,
        })

    # 汇总表
    print(f"\n{'='*60}")
    print(f"{'Image':<25s} {'N':>3s}  {'Top%':>6s}  {'Converged':>10s}")
    print(f"{'-'*60}")
    for s in summary:
        print(f"{s['image']:<25s} {s['n_materials']:>3d}  {s['top_material']*100:>5.1f}%  "
              f"{str(s['converged']):>10s}")


if __name__ == '__main__':
    main()
