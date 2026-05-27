"""
五行子网络 + 基本单元 快速测试

用 5 种基本视觉运算模拟五个子网络的输出，
看看 5 个视角的交集能不能自然地找出"物品在哪"。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def five_perspectives(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    print(f"图片: {image_path} ({w}×{h})")
    
    # ─── 五个视角（用经典 CV 模拟）───
    
    # 金：边缘（Sobel 梯度幅值）
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gold = np.sqrt(sobel_x**2 + sobel_y**2)
    gold = (gold / gold.max() * 255).astype(np.uint8)
    print(f"  金（边缘）: 强度范围 {gold.min()}~{gold.max()}")
    
    # 木：连续性（局部方向一致性 → 用 Gabor 滤波模拟）
    # 响应高的区域 = 纹理方向一致 → 连续性高
    wood = cv2.GaussianBlur(gray, (15, 15), 0)
    wood_grad_x = cv2.Sobel(wood, cv2.CV_64F, 1, 0)
    wood_grad_y = cv2.Sobel(wood, cv2.CV_64F, 0, 1)
    wood_mag = np.sqrt(wood_grad_x**2 + wood_grad_y**2)
    # 局部梯度幅值低 = 平滑连续 = 木高
    wood = 255 - (wood_mag / wood_mag.max() * 255).astype(np.uint8)
    print(f"  木（连续性）: 强度范围 {wood.min()}~{wood.max()}")
    
    # 水：深度（用亮度梯度模拟深度变化趋势）
    # 从亮到暗的渐变 = 深度的变化
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    water = cv2.Laplacian(blur, cv2.CV_64F)
    water = np.abs(water)
    water = (water / water.max() * 255).astype(np.uint8)
    print(f"  水（深度变化）: 强度范围 {water.min()}~{water.max()}")
    
    # 火：光照一致性（局部颜色方差）
    # 方差低 = 光照/颜色均匀 = 火的视角认为这里一致
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    local_std = cv2.GaussianBlur(hsv[:,:,1], (15, 15), 0)  # 饱和度局部均值
    local_std = np.abs(hsv[:,:,1].astype(float) - local_std.astype(float))
    fire = (255 - (local_std / local_std.max() * 255)).astype(np.uint8)
    print(f"  火（光照/颜色一致）: 强度范围 {fire.min()}~{fire.max()}")
    
    # 土：纹理一致（局部二值模式响应）
    # 纹理均匀处 = 土的视角认为质地一致
    lbp = cv2.GaussianBlur(gray, (31, 31), 0)
    earth_diff = np.abs(gray.astype(float) - lbp.astype(float))
    earth = (255 - (earth_diff / earth_diff.max() * 255)).astype(np.uint8)
    print(f"  土（纹理一致）: 强度范围 {earth.min()}~{earth.max()}")
    
    # ─── 交集：五个视角都高的区域 ───
    # 归一化到 0~1
    gold_n = gold.astype(float) / 255
    wood_n = wood.astype(float) / 255
    water_n = water.astype(float) / 255
    fire_n = fire.astype(float) / 255
    earth_n = earth.astype(float) / 255
    
    # 交集 = 五个都高 → 取最小值（最严格的交集）
    intersection = np.minimum.reduce([gold_n, wood_n, water_n, fire_n, earth_n])
    intersection_img = (intersection * 255).astype(np.uint8)
    
    # 加权平均（每个视角的贡献不同）
    weighted = (gold_n * 0.2 + wood_n * 0.2 + water_n * 0.2 + 
                fire_n * 0.2 + earth_n * 0.2)
    weighted_img = (weighted * 255).astype(np.uint8)
    
    # ─── 可视化 ───
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0,0].imshow(img_rgb)
    axes[0,0].set_title("Original")
    axes[0,1].imshow(gold, cmap='gray')
    axes[0,1].set_title("Metal (Edge)")
    axes[0,2].imshow(wood, cmap='gray')
    axes[0,2].set_title("Wood (Continuity)")
    axes[0,3].imshow(water, cmap='gray')
    axes[0,3].set_title("Water (Depth)")
    axes[1,0].imshow(fire, cmap='gray')
    axes[1,0].set_title("Fire (Color)")
    axes[1,1].imshow(earth, cmap='gray')
    axes[1,1].set_title("Earth (Texture)")
    axes[1,2].imshow(intersection_img, cmap='gray')
    axes[1,2].set_title("Intersection (5-vote)")
    axes[1,3].imshow(weighted_img, cmap='gray')
    axes[1,3].set_title("Weighted Average")
    
    for ax in axes.ravel():
        ax.axis('off')
    
    out_path = str(output_dir / f"five_perspectives_{Path(image_path).stem}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n结果图已保存: {out_path}")
    
    # ─── 分析交集 vs 加权 ───
    print(f"\n{'='*60}")
    print("分析：")
    
    # 交集的高亮区域占比
    high_intersection = (intersection > 0.5).sum() / (h * w) * 100
    high_weighted = (weighted > 0.5).sum() / (h * w) * 100
    print(f"  交集 > 0.5 的面积占比: {high_intersection:.1f}%")
    print(f"  加权平均 > 0.5 的面积占比: {high_weighted:.1f}%")
    
    # 五个视角的相关系数
    perspectives = [gold_n, wood_n, water_n, fire_n, earth_n]
    names = ["金", "木", "水", "火", "土"]
    print(f"\n  视角之间的相关系数:")
    for i in range(5):
        for j in range(i+1, 5):
            corr = np.corrcoef(perspectives[i].ravel(), perspectives[j].ravel())[0,1]
            print(f"    {names[i]}-{names[j]}: {corr:.3f}")
    
    print(f"\n{'='*60}")
    print("Observations:")
    print("  1. Lower correlation between views = better complementarity")
    print("  2. Intersection area = where ALL 5 views agree = object region")
    print("  3. Views with low correlation filter out each other's blind spots")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    five_perspectives(args.image)
