"""
跨图物质匹配 — 批量化报告

给 Caltech101 每类取 3 张图，计算类内 vs 类间几何签名相似度。
纯几何算子（8维），不加表象层。证明模型理解世界本质，非表象。
"""
import os, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch, cv2, numpy as np
from collections import Counter
from src.pipeline import BaguaPipeline
from src.field_core import (essence_field_compute, edge_aware_diffusion,
                             field_to_materials)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = 128
DATA = 'data/caltech101/101_ObjectCategories'
OP_NAMES = ['qian','kun','zhen','xun','kan','li','gen','dui']


def extract_geo_signature(img_path, pipe):
    """提取几何签名（纯 8 算子，无表象）"""
    img = cv2.imread(img_path)
    if img is None: return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255

    with torch.no_grad():
        f, e, _ = essence_field_compute(x, pipe, use_appearance=False)
    fs, _ = edge_aware_diffusion(f, e, n_iters=20, alpha=0.15)
    labels, _, _, prototypes = field_to_materials(f, e, n_clusters=2)

    H, W = labels.shape
    center = labels[H//3:2*H//3, W//3:2*W//3].flatten()
    best_k = Counter(center).most_common(1)[0][0]
    vec = fs[0].permute(1,2,0).reshape(-1, 8).cpu().numpy()
    return vec[labels.flatten() == best_k].mean(axis=0)


def main():
    pipe = BaguaPipeline(d=8).to(DEVICE).eval()
    categories = sorted(d for d in os.listdir(DATA)
                        if os.path.isdir(os.path.join(DATA, d))
                        and d != 'BACKGROUND_Google')

    # 每类取 3 张
    random.seed(42)
    all_sigs = {}  # {cat: [proto, proto, proto]}
    for cat in categories:
        cat_dir = os.path.join(DATA, cat)
        imgs = [f for f in os.listdir(cat_dir)
                if f.endswith(('.jpg','.jpeg','.png'))]
        if len(imgs) < 3:
            continue
        imgs = random.sample(imgs, 3)
        protos = []
        for img_name in imgs:
            proto = extract_geo_signature(os.path.join(cat_dir, img_name), pipe)
            if proto is not None:
                protos.append(proto)
        if len(protos) >= 2:
            all_sigs[cat] = protos

    print(f'{len(all_sigs)} categories with >=2 valid prototypes\n')

    # 计算每种物类别的类内稳定性和类间相似度
    cat_means = {}  # {cat: mean_proto}
    cat_within = {}  # {cat: avg_within_sim}
    for cat, protos in all_sigs.items():
        mean_p = np.mean(protos, axis=0)
        cat_means[cat] = mean_p
        # 类内成对相似度
        sims = []
        for i in range(len(protos)):
            for j in range(i+1, len(protos)):
                sim = np.dot(protos[i], protos[j]) / (
                    np.linalg.norm(protos[i]) * np.linalg.norm(protos[j]) + 1e-8)
                sims.append(sim)
        cat_within[cat] = np.mean(sims) if sims else 0

    # 全局统计
    all_within = list(cat_within.values())
    cats_list = sorted(cat_within.keys())
    N = len(cats_list)

    # 类间相似度矩阵
    sim_matrix = np.zeros((N, N))
    for i, ci in enumerate(cats_list):
        for j, cj in enumerate(cats_list):
            sim_matrix[i, j] = np.dot(cat_means[ci], cat_means[cj]) / (
                np.linalg.norm(cat_means[ci]) * np.linalg.norm(cat_means[cj]) + 1e-8)

    # 打分：每类的"辨识度" = 类内 - 类间
    identifiability = {}
    for i, ci in enumerate(cats_list):
        within = cat_within[ci]
        between = np.mean([sim_matrix[i, j] for j in range(N) if j != i])
        identifiability[ci] = within - between

    # 输出：Top 15 最容易跨图匹配的类别
    print("=== Top 15: 几何签名最稳定的类别（类内相似度最高）===")
    print(f"{'Category':<25s} {'Within':>7s} {'Between':>7s} {'Diff':>7s}  Dominant Ops")
    print("-" * 85)
    for cat, _ in sorted(cat_within.items(), key=lambda x: -x[1])[:15]:
        w = cat_within[cat]
        i = cats_list.index(cat)
        b = np.mean([sim_matrix[i, j] for j in range(N) if j != i])
        top3_idx = np.argsort(-cat_means[cat])[:3]
        top3 = '+'.join(f'{OP_NAMES[j]}' for j in top3_idx)
        print(f"{cat:<25s} {w:7.4f} {b:7.4f} {w-b:+7.4f}  {top3}")

    print(f"\n=== Bottom 15: 几何签名最不稳定的类别（类内差异大）===")
    print(f"{'Category':<25s} {'Within':>7s} {'Between':>7s} {'Diff':>7s}  Dominant Ops")
    print("-" * 85)
    for cat, _ in sorted(cat_within.items(), key=lambda x: x[1])[:15]:
        w = cat_within[cat]
        i = cats_list.index(cat)
        b = np.mean([sim_matrix[i, j] for j in range(N) if j != i])
        top3_idx = np.argsort(-cat_means[cat])[:3]
        top3 = '+'.join(f'{OP_NAMES[j]}' for j in top3_idx)
        print(f"{cat:<25s} {w:7.4f} {b:7.4f} {w-b:+7.4f}  {top3}")

    # 全局指标
    within_arr = np.array(all_within)
    between_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            between_pairs.append(sim_matrix[i, j])
    between_arr = np.array(between_pairs)

    print(f"\n=== 全局统计 ===")
    print(f"类别数: {N}")
    print(f"类内均值: {within_arr.mean():.4f}  std: {within_arr.std():.4f}")
    print(f"类间均值: {between_arr.mean():.4f}  std: {between_arr.std():.4f}")
    print(f"差值:     {within_arr.mean() - between_arr.mean():+.4f}")

    from scipy import stats
    t, p = stats.ttest_ind(within_arr, between_arr[:1000], equal_var=False)
    print(f"t={t:.2f}  p={p:.2e}  {'***显著***' if p < 0.001 else '不显著'}")

    # 辨识度分布
    idents = list(identifiability.values())
    n_pos = sum(1 for x in idents if x > 0)
    n_neg = sum(1 for x in idents if x <= 0)
    print(f"可辨识(>0): {n_pos}/{N} ({n_pos/N*100:.0f}%)")
    print(f"不可辨识(<=0): {n_neg}/{N} ({n_neg/N*100:.0f}%)")

    return cat_within, sim_matrix, cats_list


if __name__ == '__main__':
    main()
