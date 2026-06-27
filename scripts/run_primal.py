"""
Primal Field — 一键物质涌现

用法: python scripts/run_primal.py photo.jpg
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

from src.primal import primal_relax, extract_domains

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(img_path, tau=0.02, alpha=0.3, n_iters=50, repulsion=0.0):
    img = cv2.imread(img_path)
    if img is None:
        print(f'Cannot read: {img_path}')
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    img_small = cv2.resize(img_rgb, (128, 128))
    x = torch.from_numpy(img_small).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.

    # 场弛豫: 吸引 + 排斥
    field, conv = primal_relax(x, tau=tau, alpha=alpha, n_iters=n_iters,
                               repulsion=repulsion)
    labels, n_domains = extract_domains(field)

    # 多色分割
    pal = np.array([[230, 80, 80], [80, 180, 80], [80, 80, 220],
                    [220, 180, 40], [180, 80, 200], [80, 200, 200]]) / 255.
    seg = np.zeros((128, 128, 3))
    for k in range(min(6, n_domains)):
        seg[labels == k] = pal[k]
    overlay = np.clip(img_small.astype(float) / 255 * 0.35 + seg * 0.65, 0, 1)

    # 梯度边界
    phi_np = field[0].cpu().numpy()
    gy = np.abs(np.diff(phi_np, axis=1, append=phi_np[:, -1:, :])).mean(0)
    gx = np.abs(np.diff(phi_np, axis=2, append=phi_np[:, :, -1:])).mean(0)
    grad = gy + gx

    # 弛豫后图像
    phi_rgb = np.clip(field[0].permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_small); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(phi_rgb); axes[1].set_title(f'Primal Field\n({len(conv)} iters)'); axes[1].axis('off')
    axes[2].imshow(grad, cmap='hot'); axes[2].set_title('Field Gradient\n(= natural boundary)'); axes[2].axis('off')
    axes[3].imshow(overlay); axes[3].set_title(f'{n_domains} Domains'); axes[3].axis('off')

    name = os.path.splitext(os.path.basename(img_path))[0]
    out = f'test_output/primal_{name}.png'
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()

    areas = [(labels == k).sum() / (128 * 128) * 100 for k in range(n_domains)]
    print(f'Image: {H}x{W}  |  Primal field: {len(conv)} iters')
    print(f'Domains: {n_domains}')
    for k in sorted(range(n_domains), key=lambda i: -areas[i]):
        print(f'  M{k+1}: {areas[k]:.0f}%')
    print(f'Saved: {out}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Primal Field')
    parser.add_argument('image', help='图片路径')
    parser.add_argument('--tau', type=float, default=0.02, help='温度(越小越挑剔)')
    parser.add_argument('--alpha', type=float, default=0.3, help='步长')
    parser.add_argument('--iters', type=int, default=50, help='最大迭代数')
    parser.add_argument('--repulsion', type=float, default=0.0, help='排斥强度 [0,1)')
    args = parser.parse_args()
    run(args.image, args.tau, args.alpha, args.iters, args.repulsion)
