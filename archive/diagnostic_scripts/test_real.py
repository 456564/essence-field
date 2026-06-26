"""
八卦算子 → 真实照片响应分析

对 test_maccup.png 运行所有算子
观察 8 张响应图：哪里亮？哪里暗？
"""
import sys, torch, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.operators import BAGUA_OPERATORS, BaguaOperatorLayer
import matplotlib.pyplot as plt


def test():
    img_path = "test_maccup.png"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)

    op = BaguaOperatorLayer()
    with torch.no_grad():
        results = op(x)

    operator_names = list(BAGUA_OPERATORS.keys())
    operator_cn = [v[0] for v in BAGUA_OPERATORS.values()]

    # ─── 显示：原图 + 8 张响应图 ───
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # 原图
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    for i, (op_name, cn) in enumerate(zip(operator_names, operator_cn)):
        row, col = (i+1) // 3, (i+1) % 3
        heatmap = results[op_name][0, 0].cpu().numpy()
        ax = axes[row, col]
        ax.imshow(img_rgb, alpha=0.5)
        im = ax.imshow(heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        ax.set_title(op_name, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    out_path = "test_output/bagua_real_maccup.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {out_path}")

        # ─── 数据：每个算子在全图的统计 ───
    print(f"\n{'='*55}")
    print("Operator mean  max@pos")
    print(f"{'='*55}")
    for op_name, cn in zip(operator_names, operator_cn):
        hm = results[op_name][0, 0].cpu().numpy()
        mean_val = hm.mean()
        max_val = hm.max()
        max_pos = np.unravel_index(hm.argmax(), hm.shape)
        print(f"{op_name:<6} {mean_val:.3f}    {max_val:.3f} @({max_pos[0]},{max_pos[1]})")

    print(f"\nExpected response patterns for a mug:")
    print(f"  Qian(circle)   → rim area")
    print(f"  Kun(flat)      → table/background")
    print(f"  Zhen(edge)     → mug contour")
    print(f"  Kan(curve)     → rim arc")
    print(f"  Dui(gap)       → handle/gap")
    print(f"  Gen(block)     → mug body")


if __name__ == "__main__":
    test()
