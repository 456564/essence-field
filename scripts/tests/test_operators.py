"""
八卦算子完整验证

对每张合成测试图跑所有 8 个算子
量化每个算子的响应强度，看是否正确对应
"""

import sys, torch, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.operators import BAGUA_OPERATORS, BaguaOperatorLayer
import matplotlib.pyplot as plt


def test():
    # 合成测试图路径
    syn_dir = Path("test_output")
    images = {
        "qian_circle": "圆形",
        "kun_flat": "平坦",
        "zhen_edge": "边缘",
        "xun_lines": "细线",
        "kan_curve": "曲线",
        "li_color": "颜色",
        "gen_block": "块状",
        "dui_gap": "开口",
    }

    op = BaguaOperatorLayer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 存储每张图的响应
    responses = {}  # {img_name: {op_name: value}}
    operator_names = list(BAGUA_OPERATORS.keys())
    operator_cn = [v[0] for v in BAGUA_OPERATORS.values()]

    for img_key, img_cn in images.items():
        path = syn_dir / f"syn_{img_key}.png"
        if not path.exists():
            print(f"  ❌ 找不到 {path}")
            continue

        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 转 tensor
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(device)

        # 运行所有算子
        with torch.no_grad():
            results = op(x)

        responses[img_key] = {}
        for op_name in operator_names:
            # 取全图均值
            val = results[op_name].mean().item()
            responses[img_key][op_name] = val

    # ─── 输出结果表格 ───
    print(f"\n{'='*85}")
    print(f"{'算子\\输入':<10}", end="")
    for img_cn in images.values():
        print(f"{img_cn:<10}", end="")
    print()
    print(f"{'='*85}")

    for i, op_name in enumerate(operator_names):
        print(f"{operator_cn[i]:<10}", end="")
        for img_key in images.keys():
            val = responses.get(img_key, {}).get(op_name, 0)
            bar = '█' * max(0, min(int(val * 20), 20))
            print(f"{val:.3f} {bar:<6}", end="  ")
        print()

    # ─── 对角线分析：期望 vs 实际 ───
    expected = {
        "qian_circle": "qian",
        "kun_flat": "kun",
        "zhen_edge": "zhen",
        "xun_lines": "xun",
        "kan_curve": "kan",
        "li_color": "li",
        "gen_block": "gen",
        "dui_gap": "dui",
    }

    print(f"\n{'='*85}")
    print("对角线分析：每张图对应的算子是否响应最强")
    print(f"{'='*85}")
    hits = 0
    for img_key, expected_op in expected.items():
        img_responses = responses.get(img_key, {})
        if not img_responses:
            continue
        # 找实际响应最强的算子
        strongest = max(img_responses, key=img_responses.get)
        strongest_val = img_responses[strongest]
        expected_val = img_responses.get(expected_op, 0)
        is_hit = strongest == expected_op
        
        status = "✅" if is_hit else "❌"
        print(f"  {images[img_key]:6s} → 期望:{BAGUA_OPERATORS[expected_op][0]:4s} "
              f"({expected_val:.3f}) 实际最强:{BAGUA_OPERATORS[strongest][0]:4s} "
              f"({strongest_val:.3f}) {status}")
        if is_hit:
            hits += 1

    total = len(expected)
    print(f"\n  命中率: {hits}/{total} = {hits/total*100:.0f}%")

    # ─── 热力图可视化 ───
    fig, axes = plt.subplots(len(images), len(operator_names) + 1, 
                              figsize=(16, 10))
    
    for row, (img_key, img_cn) in enumerate(images.items()):
        path = syn_dir / f"syn_{img_key}.png"
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(device)
        
        with torch.no_grad():
            results = op(x)
        
        # 原图
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_ylabel(images[img_key], fontsize=9)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title("原图", fontsize=9)
        
        # 每个算子
        for col, op_name in enumerate(operator_names):
            heatmap = results[op_name][0, 0].cpu().numpy()
            axes[row, col+1].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
            axes[row, col+1].axis('off')
            if row == 0:
                axes[row, col+1].set_title(BAGUA_OPERATORS[op_name][0], fontsize=8)

    plt.suptitle("八卦算子响应矩阵：每行=输入图，每列=算子", fontsize=11)
    plt.tight_layout()
    out_path = "test_output/operator_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n热力图矩阵已保存: {out_path}")


if __name__ == "__main__":
    test()
