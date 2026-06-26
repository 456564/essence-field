"""
合成测试图：每张图对应一个卦的"纯粹象"

用于验证 8 卦子网络的先天偏向是否正确。
"""

import cv2
import numpy as np
from pathlib import Path


def generate_test_images(output_dir="test_output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    size = 224
    
    # ─── 乾（天/圆）───
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    cv2.circle(img, (112, 112), 70, (60, 60, 60), -1)
    cv2.circle(img, (112, 112), 60, (80, 80, 80), -1)
    cv2.imwrite(str(output_dir / "syn_qian_circle.png"), img)
    
    # ─── 坤（地/平坦）───
    img = np.ones((size, size, 3), dtype=np.uint8) * 180
    # 渐变非常平缓
    for y in range(size):
        val = 160 + int(y / size * 30)
        img[y, :] = [val, val, val]
    cv2.imwrite(str(output_dir / "syn_kun_flat.png"), img)
    
    # ─── 震（雷/边缘）───
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (50, 50), (174, 174), (30, 30, 30), -1)
    cv2.rectangle(img, (60, 60), (164, 164), (200, 200, 200), -1)  # 高对比边缘
    cv2.imwrite(str(output_dir / "syn_zhen_edge.png"), img)
    
    # ─── 巽（风/细长）───
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    for i in range(10):
        x = 20 + i * 20
        cv2.line(img, (x, 30), (x + 15, 194), (50, 50, 50), 2)
    cv2.imwrite(str(output_dir / "syn_xun_lines.png"), img)
    
    # ─── 坎（水/渐变弯曲）───
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    for x in range(size):
        y1 = int(112 + 60 * np.sin(x * 0.05))
        y2 = int(112 + 60 * np.sin(x * 0.05 + np.pi))
        cv2.circle(img, (x, y1), 3, (60, 60, 60), -1)
        cv2.circle(img, (x, y2), 3, (100, 100, 100), -1)
    cv2.imwrite(str(output_dir / "syn_kan_curve.png"), img)
    
    # ─── 离（火/颜色）───
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 60, (0, 0, 255), -1)      # 红
    cv2.circle(img, (80, 80), 25, (0, 255, 255), -1)       # 黄
    cv2.circle(img, (144, 80), 20, (255, 0, 255), -1)      # 紫
    cv2.imwrite(str(output_dir / "syn_li_color.png"), img)
    
    # ─── 艮（山/块状）───
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (40, 40), (90, 184), (60, 60, 60), -1)
    cv2.rectangle(img, (136, 60), (184, 184), (80, 80, 80), -1)
    cv2.rectangle(img, (80, 100), (136, 184), (70, 70, 70), -1)
    cv2.imwrite(str(output_dir / "syn_gen_block.png"), img)
    
    # ─── 兑（泽/开口凹陷）───
    img = np.ones((size, size, 3), dtype=np.uint8) * 180
    cv2.circle(img, (112, 112), 70, (60, 60, 60), -1)
    # 开口
    cv2.rectangle(img, (112-15, 112-75), (112+15, 112-30), (180, 180, 180), -1)
    cv2.imwrite(str(output_dir / "syn_dui_gap.png"), img)
    
    print(f"生成 8 张合成测试图到 {output_dir}/")
    names = ["qian_circle", "kun_flat", "zhen_edge", "xun_lines",
             "kan_curve", "li_color", "gen_block", "dui_gap"]
    for n in names:
        print(f"  syn_{n}.png")


if __name__ == "__main__":
    generate_test_images()
