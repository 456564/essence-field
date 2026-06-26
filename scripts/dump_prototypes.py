"""导出每张图的主物质算子强度"""
import os, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch, cv2, numpy as np
from src.pipeline import BaguaPipeline
from src.field_core import (essence_field_compute, edge_aware_diffusion,
                             field_to_materials)
from src.operators import BASE_OPS

TARGET_CATEGORIES = [
    'cup', 'airplanes', 'Motorbikes', 'watch', 'chair',
    'dolphin', 'soccer_ball', 'butterfly', 'laptop', 'revolver',
]
SAMPLES = 5
SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OP_NAMES = ['qian','kun','zhen','xun','kan','li','gen','dui']


def extract_profile(img_path, pipe):
    img = cv2.imread(img_path)
    if img is None: return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    x = torch.from_numpy(img_small).permute(2,0,1).float()
    x = x.unsqueeze(0).to(DEVICE) / 255.0

    with torch.no_grad():
        field_8, edge_metric, _ = essence_field_compute(
            x, pipe, presmooth_sigma=3.0, edge_scale=5.0)
    field_smooth, _ = edge_aware_diffusion(
        field_8, edge_metric, n_iters=20, alpha=0.15)
    labels, _, _, prototypes = field_to_materials(
        field_8, edge_metric, n_clusters=2)

    raw_vec = field_smooth[0].permute(1,2,0).reshape(-1,8).cpu().numpy()
    labels_f = labels.flatten()
    best_k = max(range(len(prototypes)), key=lambda k: (labels_f == k).sum())
    proto_raw = raw_vec[labels_f == best_k].mean(axis=0)
    return proto_raw


def main():
    data_root = 'data/caltech101/101_ObjectCategories'
    pipe = BaguaPipeline(d=8).to(DEVICE).eval()

    all_data = {}  # {cat: [(img_name, [8]), ...]}

    for cat in TARGET_CATEGORIES:
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        imgs = [f for f in os.listdir(cat_dir) if f.endswith(('.jpg','.jpeg','.png'))]
        random.seed(42)
        imgs = random.sample(imgs, min(SAMPLES, len(imgs)))
        cat_data = []
        for img_name in sorted(imgs):
            proto = extract_profile(os.path.join(cat_dir, img_name), pipe)
            if proto is not None:
                cat_data.append((img_name, proto))
        all_data[cat] = cat_data

    # === 逐张打印 ===
    print(f"{'image':<40s} {'qian':>6s} {'kun':>6s} {'zhen':>6s} {'xun':>6s} {'kan':>6s} {'li':>6s} {'gen':>6s} {'dui':>6s}  top3")
    print("-" * 125)
    for cat in TARGET_CATEGORIES:
        if cat not in all_data: continue
        for img_name, proto in all_data[cat]:
            nums = ' '.join(f'{proto[i]:6.3f}' for i in range(8))
            top3_idx = np.argsort(-proto)[:3]
            top3_str = '>'.join(f'{OP_NAMES[i]}({proto[i]:.2f})' for i in top3_idx)
            print(f'{cat}/{img_name:<30s} {nums}  | {top3_str}')
        print("-" * 125)

    # === 类别汇总 ===
    print(f"\n{'category':<15s} {'qian':>7s} {'kun':>7s} {'zhen':>7s} {'xun':>7s} {'kan':>7s} {'li':>7s} {'gen':>7s} {'dui':>7s}  consistency")
    print("-" * 125)
    for cat in TARGET_CATEGORIES:
        if cat not in all_data: continue
        protos = np.array([p for _, p in all_data[cat]])  # [N, 8]
        mean = protos.mean(axis=0)
        std = protos.std(axis=0)
        # 一致性 = 主导算子的稳定性（top1 算子跨样本的频率）
        top1_ops = [OP_NAMES[np.argmax(p)] for p in protos]
        from collections import Counter
        top1_counts = Counter(top1_ops)
        best_op, best_count = top1_counts.most_common(1)[0]
        consistency = f'{best_op}({best_count}/{len(protos)})'

        nums = ' '.join(f'{mean[i]:7.3f}' for i in range(8))
        print(f'{cat:<15s} {nums}  {consistency}')


if __name__ == '__main__':
    main()
