"""
二图物质对比 — 是否同一种物质？

用法:
  python scripts/compare.py cup1.jpg cup2.jpg
  python scripts/compare.py cup.jpg airplane.jpg
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch, cv2, numpy as np
from src.pipeline import BaguaPipeline
from src.field_core import (essence_field_compute, edge_aware_diffusion,
                             field_to_materials)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 128
OP_NAMES = ['qian','kun','zhen','xun','kan','li','gen','dui']


def extract_signature(img_path, pipe):
    """提取图片的主物质算子签名"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255

    with torch.no_grad():
        field_8, edge_metric, _ = essence_field_compute(
            x, pipe, presmooth_sigma=3.0, edge_scale=5.0,
            use_appearance=False, use_global_context=False)
    field_smooth, _ = edge_aware_diffusion(
        field_8, edge_metric, n_iters=20, alpha=0.15)
    labels, _, _, prototypes = field_to_materials(
        field_8, edge_metric, n_clusters=2)

    # 取图像中心的簇（物体通常在中心，背景在边缘）
    C = field_smooth.shape[1]
    vec = field_smooth[0].permute(1,2,0).reshape(-1, C).cpu().numpy()
    H, W = labels.shape
    cy, cx = H // 2, W // 2
    # 中心 40% 区域的像素索引
    y0, y1 = int(H * 0.3), int(H * 0.7)
    x0, x1 = int(W * 0.3), int(W * 0.7)
    labels_f = labels.flatten()
    center_labels = labels[y0:y1, x0:x1].flatten()

    # 选中心最多的簇
    from collections import Counter
    center_counts = Counter(center_labels)
    best_k = center_counts.most_common(1)[0][0]
    proto = vec[labels_f == best_k].mean(axis=0)

    return proto


def compare(img_a, img_b, pipe):
    """对比两张图的主物质"""
    sig_a = extract_signature(img_a, pipe)
    sig_b = extract_signature(img_b, pipe)

    if sig_a is None or sig_b is None:
        return None

    # 余弦相似度
    sim = np.dot(sig_a, sig_b) / (np.linalg.norm(sig_a) * np.linalg.norm(sig_b) + 1e-8)

    # Top 3 算子重叠
    top_a = set(np.argsort(-sig_a)[:3])
    top_b = set(np.argsort(-sig_b)[:3])
    overlap = len(top_a & top_b) / 3

    return {
        'similarity': float(sim),
        'top3_overlap': overlap,
        'sig_a': sig_a,
        'sig_b': sig_b,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='二图物质对比')
    parser.add_argument('image_a', help='图片A路径')
    parser.add_argument('image_b', help='图片B路径')
    args = parser.parse_args()

    pipe = BaguaPipeline(d=8).to(DEVICE).eval()
    result = compare(args.image_a, args.image_b, pipe)

    if result is None:
        print("ERROR: 无法读取图片")
        return

    sim = result['similarity']
    overlap = result['top3_overlap']

    # 判定
    if sim > 0.85 and overlap >= 0.67:
        verdict = "同一种物质"
    elif sim > 0.7:
        verdict = "可能同一种物质（相似度高）"
    elif sim < 0.3:
        verdict = "不同物质"
    else:
        verdict = "不确定（几何签名有部分重叠）"

    print(f"A: {args.image_a}")
    print(f"B: {args.image_b}")
    print(f"\n相似度: {sim:.4f}")
    print(f"Top-3 算子重叠: {overlap:.0%}")
    print(f"判定: {verdict}")
    print(f"\nA 签名: {_format_sig(result['sig_a'])}")
    print(f"B 签名: {_format_sig(result['sig_b'])}")


def _format_sig(sig):
    top3 = np.argsort(-sig)[:3]
    parts = [f'{OP_NAMES[i]}={sig[i]:.3f}' for i in top3]
    return ', '.join(parts)


if __name__ == '__main__':
    main()
