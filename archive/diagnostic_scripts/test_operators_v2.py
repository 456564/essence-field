"""
八卦算子精确验证 — 只测前景区域（排除背景干扰）

每张合成图 = 均匀背景 + 前景图案
只在前景区域内计算算子响应 → 真正的选择性
"""

import sys, torch, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.operators import BAGUA_OPERATORS, BaguaOperatorLayer


def get_foreground_mask(gray_img, threshold=30):
    """
    通过亮度阈值分离前景和背景。
    合成图：背景 200，前景 60 → 差值大
    """
    # 灰度图
    if gray_img.ndim == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray_img
    
    # 找到偏离均值最多的区域
    mean_val = gray.mean()
    mask = np.abs(gray.astype(float) - mean_val) > threshold
    return mask


def test():
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
    operator_names = list(BAGUA_OPERATORS.keys())
    operator_cn = [v[0] for v in BAGUA_OPERATORS.values()]

    print(f"\n{'='*100}")
    print(f"{'输入':<8}", end="")
    for cn in operator_cn:
        print(f"{cn+'前景':<12}", end="")
    print(f"{'坤背景':<10}{'说明':<20}")
    print(f"{'='*100}")

    # 存储每张图的"最佳算子"
    best_ops = {}

    for img_key, img_cn in images.items():
        path = syn_dir / f"syn_{img_key}.png"
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 获取前景mask
        mask = get_foreground_mask(img_rgb, threshold=25)
        # 膨胀一下，包括边缘区域
        kernel = np.ones((5,5), dtype=np.uint8)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        background = (~mask_dilated).astype(bool)

        # 转 tensor
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            results = op(x)

        # 提取前景/背景区域的响应
        n_fg = float(mask_dilated.sum()) + 1e-6
        n_bg = float(background.sum()) + 1e-6

        print(f"{img_cn:<8}", end="")
        best_val = -1
        best_name = ""
        kun_bg_val = 0

        for op_name in operator_names:
            heatmap = results[op_name][0, 0].cpu().numpy()
            
            # 前景区域均值
            fg_val = float((heatmap * mask_dilated).sum()) / n_fg
            # 背景区域均值（仅坤）
            if op_name == "kun":
                kun_bg_val = float((heatmap * background).sum()) / n_bg
            
            print(f"{fg_val:.4f}    ", end="")
            
            if fg_val > best_val:
                best_val = fg_val
                best_name = BAGUA_OPERATORS[op_name][0]

        print(f"  {kun_bg_val:.4f}      ", end="")
        
        # 期望的算子
        expected = images[img_key]
        # 找实际最强的
        expected_op = {
            "qian_circle": "乾天", "kun_flat": "坤地", "zhen_edge": "震雷",
            "xun_lines": "巽风", "kan_curve": "坎水", "li_color": "离火",
            "gen_block": "艮山", "dui_gap": "兑泽",
        }[img_key]
        status = "✅" if best_name == expected_op else "❌"
        print(f"期望:{expected_op} 实际:{best_name} {status}")

    print(f"{'='*100}")
    print("说明：前景区域=图案本身（排除均匀背景）")
    print("     坤背景=坤算子在背景区域的响应（应≈0）")


if __name__ == "__main__":
    test()
