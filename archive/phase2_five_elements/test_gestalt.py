"""
格式塔原则分组测试

不用边缘检测，用邻近性+相似性把像素分组。
看看不依赖边缘能不能分出"东西在哪"。
"""

import numpy as np
import cv2
from pathlib import Path
from skimage.segmentation import slic, mark_boundaries
from skimage.graph import rag_mean_color
from skimage import graph as skgraph


def gestalt_grouping(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    print(f"图片: {image_path} ({w}×{h})")

    # ─── 第一步：超像素分割（过分割）───
    # 把图切成很多小片，每片内部颜色相似
    # 相当于"邻近性 + 相似性"的初始分组
    n_segments = 200
    segments = slic(
        img_rgb, n_segments=n_segments, compactness=10, 
        start_label=0, channel_axis=-1
    )
    n_labels = len(np.unique(segments))
    print(f"超像素数: {n_labels}（目标 {n_segments}）")

    # ─── 第二步：根据相似性合并超像素 ───
    # RAG = Region Adjacency Graph（区域邻接图）
    # 相邻的超像素之间计算颜色相似度
    # 相似度高的合并 → 更大的区域
    g = rag_mean_color(img_rgb, segments)

    # 用不同的合并阈值，看效果
    def dummy_merge(*args):
        pass
    def dummy_weight(graph, src, dst, n):
        return {"weight": graph[src][dst]["weight"]}

    for threshold in [0.1, 0.2, 0.3, 0.5]:
        merged = skgraph.merge_hierarchical(
            segments, g,
            thresh=threshold,
            rag_copy=False,
            in_place_merge=False,
            merge_func=dummy_merge,
            weight_func=dummy_weight,
        )
        n_regions = len(np.unique(merged))
        
        # 统计每个区域的大小
        region_sizes = []
        for label in np.unique(merged):
            mask = merged == label
            size = mask.sum()
            region_sizes.append(size)
        region_sizes.sort(reverse=True)
        
        # 最大的几个区域
        top_sizes = [f"{s/(w*h)*100:.1f}%" for s in region_sizes[:5]]
        
        print(f"  阈值 {threshold:.1f}: {n_regions} 个区域 → 最大5个: {', '.join(top_sizes)}")

    # ─── 对比：传统边缘检测 ───
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\n对比：Canny 边缘检测 → {len(contours)} 个轮廓")

    # ─── 保存最佳结果 ───
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 用一个适中的阈值保存结果图
    best_thresh = 0.2
    merged_final = skgraph.merge_hierarchical(
        segments, g,
        thresh=best_thresh,
        rag_copy=False,
        in_place_merge=False,
        merge_func=dummy_merge,
        weight_func=dummy_weight,
    )
    
    # 画出区域边界
    result = mark_boundaries(img_rgb / 255.0, merged_final)
    result = (result * 255).astype(np.uint8)
    out_path = str(output_dir / f"gestalt_{Path(image_path).stem}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"\n格式塔分组结果已保存: {out_path}")

    # ─── 观察结论 ───
    print(f"\n{'='*60}")
    print("观察：")
    print("  1. 不同阈值下区域数量变化很大")
    print("  2. 太小的阈值 → 过碎（跟纹理走）")
    print("  3. 太大的阈值 → 过粗（不同物体粘一起）")
    print("  4. 没有正确的阈值——取决于图本身")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    gestalt_grouping(args.image)
