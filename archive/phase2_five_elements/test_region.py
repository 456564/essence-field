"""
最简区域检测测试

输入一张实物照片 → 检测图上有几个区域 → 区分物品和背景
目的是看最底层用已有工具会卡在哪。
"""

import cv2
import numpy as np
from pathlib import Path


def test_region_detection(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    
    print(f"图片: {image_path}")
    print(f"尺寸: {w}×{h}")
    
    # ─── 方法1：Canny 边缘检测 ───
    edges = cv2.Canny(gray, 50, 150)
    
    # ─── 方法2：边缘 → 轮廓 ───
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按面积排序，取最大的前 N 个
    areas = [(i, cv2.contourArea(c)) for i, c in enumerate(contours)]
    areas.sort(key=lambda x: -x[1])
    
    print(f"\nCanny 边缘检测后找到的轮廓数: {len(contours)}")
    
    # 分析轮廓层级
    if hierarchy is not None:
        num_parents = sum(1 for h in hierarchy[0] if h[3] == -1)  # 最外层轮廓
        print(f"最外层轮廓（疑似独立物体）: {num_parents}")
        
        # 显示最大几个轮廓
        print(f"\n最大轮廓 TOP 10（按面积排序）:")
        print(f"{'排名':<4} {'面积(pixel)':<14} {'占总图%':<10} {'中心位置':<20} {'层级关系':<10}")
        print("-" * 60)
        for rank, (i, area) in enumerate(areas[:10]):
            x, y, w_c, h_c = cv2.boundingRect(contours[i])
            center = f"({x+w_c//2}, {y+h_c//2})"
            pct = area / (w * h) * 100
            parent = hierarchy[0][i][3]  # 父轮廓索引
            print(f"{rank+1:<4} {area:<14.0f} {pct:<10.2f} {center:<20} {'子' if parent != -1 else '外层':<10}")
    
    # ─── 测试不同参数的效果 ───
    print(f"\n=== 参数敏感性测试 ===")
    for threshold in [(30, 90), (50, 150), (100, 200), (150, 300)]:
        e = cv2.Canny(gray, threshold[0], threshold[1])
        c, _ = cv2.findContours(e, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        outer = sum(1 for h in cv2.findContours(e, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1][0] if h[3] == -1)
        print(f"  Canny({threshold[0]}, {threshold[1]}) → 轮廓 {len(c)} 个, 外层 {outer} 个")
    
    # ─── 保存结果 ───
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 在原图上画轮廓
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    out_path = str(output_dir / f"contours_{Path(image_path).stem}.jpg")
    cv2.imwrite(out_path, result)
    print(f"\n轮廓图已保存: {out_path}")
    
    # ─── 观察结论 ───
    print(f"\n{'='*60}")
    print("初步观察:")
    print(f"  - 图上物体越多, 轮廓越乱")
    print(f"  - 背景纹理复杂时, 边缘检测会碎")
    print(f"  - Canny 参数对结果影响很大")
    print(f"  - 同一物体可能被拆成多个轮廓")
    print(f"  - 多个物体可能被连成一个轮廓（颜色相近时）")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    test_region_detection(args.image)
