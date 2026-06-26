"""
MVE: 单图物质自发现

验证本质场架构核心假设：
  8算子 → 64维场 → 场自组织扩散 → 物质自然分组

输入: test_maccup.png (马克杯 + 蓝色背景)
输出: 可视化面板（原图、算子响应、扩散前后对比、收敛曲线）

两种模式:
  --random: 随机权重（验证扩散机制本身是否有效）
  --ckpt PATH: 加载训练权重
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from src.pipeline import BaguaPipeline
from src.diffusion import FieldDiffusion, bilateral_diffusion
from src.operators import BASE_OPS, BAGUA_NAMES

GUA_SYMBOLS = ['乾', '坤', '震', '巽', '坎', '离', '艮', '兑']
GUA_COLORS = ['#E74C3C', '#2ECC71', '#3498DB', '#F1C40F',
              '#1ABC9C', '#E91E63', '#E67E22', '#9B59B6']


def load_image(path, size=224):
    """加载并预处理图像"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img_rgb.copy()
    img_resized = cv2.resize(img_rgb, (size, size))
    x = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return x, img_resized, original


def load_checkpoint(pipe, ckpt_path, device):
    """加载训练权重"""
    ckpt = torch.load(ckpt_path, map_location=device)
    pipe.fusion.A.data = ckpt['A']
    pipe.operator_layer.projections.load_state_dict(ckpt['proj'])
    epoch = ckpt.get('epoch', '?')
    return epoch


def compute_field_stats(field_64):
    """计算场的关键统计量"""
    B, C, H, W = field_64.shape
    flat = field_64.view(C, -1).t()  # [H*W, 64]

    # 范数统计
    norms = flat.norm(dim=1)  # [H*W]
    norm_cv = norms.std() / (norms.mean() + 1e-8)  # 变异系数

    # 方向统计
    directions = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    # 像素对之间的平均余弦相似度（采样）
    n_sample = min(2000, H * W)
    idx = torch.randperm(H * W)[:n_sample]
    sim_matrix = torch.mm(directions[idx], directions[idx].t())
    # 上三角（排除对角线）
    triu_idx = torch.triu_indices(n_sample, n_sample, offset=1)
    avg_sim = sim_matrix[triu_idx[0], triu_idx[1]].mean().item()

    return {
        'norm_mean': norms.mean().item(),
        'norm_std': norms.std().item(),
        'norm_cv': norm_cv.item(),
        'avg_pairwise_sim': avg_sim,
    }


def plot_mve_panel(img_np, field_raw, field_diff, ops, convergence,
                   save_path, stats_before=None, stats_after=None):
    """主可视化面板"""
    if stats_before is None:
        stats_before = compute_field_stats(field_raw)
    if stats_after is None:
        stats_after = compute_field_stats(field_diff)

    # 转 numpy
    field_before_np = field_raw.cpu()
    field_after_np = field_diff.cpu()

    # ═══════════════════════════════════════
    # 绘制 4×3 面板
    # ═══════════════════════════════════════
    fig = plt.figure(figsize=(20, 24))

    # --- Row 0: 原图 + 算子响应概况 ---
    ax_img = fig.add_subplot(5, 4, 1)
    ax_img.imshow(img_np)
    ax_img.set_title("原图", fontsize=11)
    ax_img.axis('off')

    # 8 算子原始响应
    for i, name in enumerate(BASE_OPS):
        ax = fig.add_subplot(5, 4, 2 + i)
        v = ops[name][0, 0].cpu().numpy()
        im = ax.imshow(v, cmap='hot',
                       vmin=np.percentile(v, 2),
                       vmax=np.percentile(v, 98))
        ax.set_title(f"{GUA_SYMBOLS[i]} {BAGUA_NAMES[name]}", fontsize=9)
        ax.axis('off')

    # --- Row 1: 扩散前 ---
    row_labels = ['扩散前', '扩散后']
    fields_np = [field_before_np, field_after_np]
    stats_list = [stats_before, stats_after]

    for row_idx, (label, fnp, stats) in enumerate(zip(row_labels, fields_np, stats_list)):
        base_col = 1 + row_idx * 2  # 1 or 3

        # 场范数
        ax_norm = fig.add_subplot(5, 4, base_col)
        norms = fnp[0].norm(dim=0).numpy()
        ax_norm.imshow(norms, cmap='hot',
                       vmin=np.percentile(norms, 2),
                       vmax=np.percentile(norms, 98))
        ax_norm.set_title(f"{label}\n范数 CV={stats['norm_cv']:.3f}", fontsize=10)
        ax_norm.axis('off')

        # 范数中位数分割
        ax_mask = fig.add_subplot(5, 4, base_col + 1)
        median = np.median(norms)
        mask = (norms > median).astype(np.float32)
        ax_mask.imshow(mask, cmap='gray')
        fg_mean = norms[mask > 0].mean()
        bg_mean = norms[mask == 0].mean()
        sep = (fg_mean - bg_mean) / (bg_mean + 1e-8)
        ax_mask.set_title(f"{label}\n分离度={sep:.1f}", fontsize=10)
        ax_mask.axis('off')

        # 最强卦 argmax 复合图
        ax_argmax = fig.add_subplot(5, 4, base_col + 2)
        from src.visualize import argmax_gua_composite
        comp = argmax_gua_composite(fnp, img_np)
        ax_argmax.imshow(comp)
        ax_argmax.set_title(f"{label} argmax复合", fontsize=10)
        ax_argmax.axis('off')

        # Blended 混合图
        ax_blend = fig.add_subplot(5, 4, base_col + 3)
        from src.visualize import blended_gua_response
        blend = blended_gua_response(fnp)
        ax_blend.imshow(blend)
        ax_blend.set_title(f"{label} blended混合", fontsize=10)
        ax_blend.axis('off')

    # --- Row 2: 单卦空间分布（扩散后）---
    for i, name in enumerate(BASE_OPS):
        ax = fig.add_subplot(5, 4, 9 + i)
        block = field_after_np[0, i*8:(i+1)*8, :, :]
        spat = block.norm(dim=0).numpy()
        im = ax.imshow(spat, cmap='hot',
                       vmin=np.percentile(spat, 2),
                       vmax=np.percentile(spat, 98))
        ax.set_title(f"{GUA_SYMBOLS[i]} {BAGUA_NAMES[name]}", fontsize=9)
        ax.axis('off')

    # --- Row 3: 收敛曲线 + 统计对比 ---
    ax_conv = fig.add_subplot(5, 2, 9)
    if convergence:
        ax_conv.plot(convergence, 'b-o', markersize=4)
        ax_conv.set_xlabel("迭代")
        ax_conv.set_ylabel("相对变化 δ")
        ax_conv.set_title(f"扩散收敛曲线 (最终 δ={convergence[-1]:.6f})")
        ax_conv.grid(True, alpha=0.3)
    else:
        ax_conv.text(0.5, 0.5, "无扩散数据", ha='center', va='center')

    # 统计对比文本
    ax_stat = fig.add_subplot(5, 2, 10)
    ax_stat.axis('off')
    stat_text = (
        f"扩散前:\n"
        f"  范数均值: {stats_before['norm_mean']:.3f}\n"
        f"  范数CV:   {stats_before['norm_cv']:.3f}\n"
        f"  平均成对相似度: {stats_before['avg_pairwise_sim']:.4f}\n\n"
        f"扩散后:\n"
        f"  范数均值: {stats_after['norm_mean']:.3f}\n"
        f"  范数CV:   {stats_after['norm_cv']:.3f}\n"
        f"  平均成对相似度: {stats_after['avg_pairwise_sim']:.4f}\n\n"
        f"CV 变化: {(stats_after['norm_cv'] - stats_before['norm_cv']) / (stats_before['norm_cv'] + 1e-8):+.1%}\n"
        f"相似度变化: {stats_after['avg_pairwise_sim'] - stats_before['avg_pairwise_sim']:+.4f}"
    )
    ax_stat.text(0.05, 0.95, stat_text, transform=ax_stat.transAxes,
                 fontsize=10, fontfamily='Microsoft YaHei', va='top')

    plt.suptitle("本质场自组织扩散 — MVE 验证", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(save_path) or 'test_output', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 面板已保存: {save_path}")

    # 打印关键指标
    print(f"\n=== 扩散诊断 ===")
    print(f"  扩散前 范数CV: {stats_before['norm_cv']:.4f}")
    print(f"  扩散后 范数CV: {stats_after['norm_cv']:.4f}")
    print(f"  扩散前 成对相似度: {stats_before['avg_pairwise_sim']:.4f}")
    print(f"  扩散后 成对相似度: {stats_after['avg_pairwise_sim']:.4f}")
    if convergence:
        print(f"  收敛: {len(convergence)} 轮, 最终 δ={convergence[-1]:.6f}")

    return stats_before, stats_after, convergence


def main():
    parser = argparse.ArgumentParser(description='本质场 MVE 验证')
    parser.add_argument('--image', default='test_maccup.png', help='输入图片路径')
    parser.add_argument('--size', type=int, default=128, help='resize 大小（默认128，避免OOM）')
    parser.add_argument('--ckpt', default=None, help='checkpoint 路径（可选）')
    parser.add_argument('--raw', action='store_true', help='原始模式：算子外积构造64维场（跳过随机权重）')
    parser.add_argument('--no-diffusion', action='store_true', help='禁用扩散（只跑原始场）')
    parser.add_argument('--bilateral', action='store_true', help='使用双边滤波扩散')
    parser.add_argument('--tau', type=float, default=0.1, help='扩散温度')
    parser.add_argument('--alpha', type=float, default=0.3, help='扩散步长')
    parser.add_argument('--n-iters', type=int, default=5, help='扩散轮数')
    parser.add_argument('--output', default='test_output/mve_panel.png', help='输出路径')
    parser.add_argument('--save-field', action='store_true', help='保存扩散后场的 .npy 文件')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载图片
    x, img_np, original = load_image(args.image, args.size)
    print(f"图片: {args.image} → {args.size}×{args.size}")

    # 创建 pipeline
    pipe = BaguaPipeline(d=8).to(device).eval()

    # 加载 checkpoint（如有）
    if args.ckpt:
        epoch = load_checkpoint(pipe, args.ckpt, device)
        print(f"Checkpoint: epoch {epoch}")
    else:
        print("[WARN] 无checkpoint -- 使用随机权重。扩散机制仍可验证。")

    # 设置扩散
    if args.no_diffusion:
        pipe.diffusion = None
        print("扩散: 关闭")
    elif args.bilateral:
        from functools import partial
        pipe.diffusion = None  # 手动调用
        print(f"扩散: 双边滤波 (σ_spatial=3, σ_range={args.tau})")
    else:
        pipe.diffusion = FieldDiffusion(
            tau=args.tau, alpha=args.alpha,
            n_iters=args.n_iters, kernel_size=5
        ).to(device)
        print(f"扩散: tau={args.tau}, alpha={args.alpha}, n_iters={args.n_iters}")

    # 运行
    x = x.to(device)
    with torch.no_grad():
        if args.raw:
            # 原始模式：跳过投影和 A 矩阵
            # 直接用算子外积构造 64 维场
            raw_ops = pipe.operator_layer.base_ops(x)
            op_list = [raw_ops[name] for name in BASE_OPS]  # 8×[B,1,H,W]
            ops_dict = raw_ops

            B, _, H, W = op_list[0].shape
            field_raw = torch.zeros(B, 64, H, W, device=device)
            for i in range(8):
                for j in range(8):
                    field_raw[:, i*8+j, :, :] = (op_list[i] * op_list[j]).squeeze(1)
            # 归一化稳定数值范围
            field_raw = F.instance_norm(field_raw.view(B, 64, -1)).view(B, 64, H, W)
        else:
            field_raw, ops_dict = pipe(x, return_operators=True)

        if args.bilateral:
            field_diff, convergence = bilateral_diffusion(
                field_raw, ops_dict, n_iters=args.n_iters,
                spatial_sigma=3.0, range_sigma=args.tau
            )
        elif pipe.diffusion is not None:
            field_diff, convergence = pipe.diffusion(field_raw, ops_dict)
        else:
            field_diff = field_raw
            convergence = []

    # 统计
    stats_before = compute_field_stats(field_raw)
    stats_after = compute_field_stats(field_diff)

    # 可视化
    plot_mve_panel(
        img_np, field_raw, field_diff, ops_dict, convergence,
        save_path=args.output,
        stats_before=stats_before, stats_after=stats_after
    )

    # 保存场数据
    if args.save_field:
        np_path = args.output.replace('.png', '_field.npy')
        np.save(np_path, field_diff.cpu().numpy())
        print(f"场数据: {np_path}")

    # MVE 判定
    print(f"\n=== MVE 判定 ===")
    cv_before = stats_before['norm_cv']
    cv_after = stats_after['norm_cv']

    # 标准1: 扩散后 CV 应下降（同物质内部趋于一致）
    if cv_after < cv_before * 0.7:
        print(f"[PASS] 范数CV 下降 {(1 - cv_after/cv_before)*100:.0f}% (阈值 30%)")
    else:
        print(f"[WARN] 范数CV 仅下降 {(1 - cv_after/cv_before)*100:.0f}% (阈值 30%)")

    # 标准2: 扩散后成对相似度应上升
    sim_delta = stats_after['avg_pairwise_sim'] - stats_before['avg_pairwise_sim']
    if sim_delta > 0.05:
        print(f"[PASS] 成对相似度上升 {sim_delta:+.4f} (阈值 +0.05)")
    else:
        print(f"[WARN] 成对相似度变化 {sim_delta:+.4f} (阈值 +0.05)")

    # 标准3: 收敛
    if convergence and convergence[-1] < 0.01:
        print(f"[PASS] 扩散收敛 (最终 delta={convergence[-1]:.6f} < 0.01)")
    elif convergence:
        print(f"[WARN] 扩散未完全收敛 (最终 delta={convergence[-1]:.6f})")

    print(f"\n结果: {args.output}")


if __name__ == '__main__':
    main()
