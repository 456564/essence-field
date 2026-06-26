"""
本质基元发现

从扩散后的 64 维场中提取离散基元（物质原型向量）。
每个基元 = 一种物质在该图像中的数学签名。

支持:
  - K-means 聚类（速度优先）
  - 密度峰值聚类 (DPC)（自动发现聚类数）
  - 跨图基元匹配（Hungarian 算法）
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def discover_primitives(field_64, method='kmeans', n_clusters=None,
                        min_clusters=2, max_clusters=8):
    """
    从本质场中发现基元。

    Args:
        field_64: [B, 64, H, W] 扩散后的本质场
        method: 'kmeans' | 'dpc'
        n_clusters: 聚类数（None = 自动选择）
        min_clusters, max_clusters: 自动选择时的范围

    Returns:
        labels: [H, W] 每个像素的基元归属
        prototypes: [K, 64] 基元原型向量
        scores: dict 聚类质量指标
    """
    B, C, H, W = field_64.shape
    # 转置并平铺 → [N, 64]
    vectors = field_64[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    # L2 归一化：InfoNCE 优化方向一致性，用方向聚类
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    if method == 'dpc':
        labels, prototypes, scores = _dpc_clustering(vectors, n_clusters)
    else:
        labels, prototypes, scores = _kmeans_clustering(
            vectors, n_clusters, min_clusters, max_clusters
        )

    labels = labels.reshape(H, W)
    return labels, prototypes, scores


def _kmeans_clustering(vectors, n_clusters, min_clusters, max_clusters):
    """K-means + 肘部法则自动选 K"""
    from sklearn.cluster import KMeans

    if n_clusters is not None:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        prototypes = km.cluster_centers_
        inertias = None
    else:
        # 自动选 K：找惯性下降的肘部
        inertias = []
        models = []
        for k in range(min_clusters, max_clusters + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_k = km.fit_predict(vectors)
            inertias.append(km.inertia_)
            models.append((km, labels_k))

        # 肘部检测：最大曲率点
        if len(inertias) >= 3:
            curves = []
            for i in range(1, len(inertias) - 1):
                # 二阶差分近似曲率
                d1 = inertias[i-1] - inertias[i]
                d2 = inertias[i] - inertias[i+1]
                curves.append(d1 - d2)
            best_k = min_clusters + np.argmax(curves)
        else:
            best_k = min_clusters

        km, labels = models[best_k - min_clusters]
        prototypes = km.cluster_centers_

    # 质量分数
    scores = _cluster_scores(vectors, labels, prototypes)
    if inertias is not None:
        scores['inertias'] = inertias

    return labels, prototypes, scores


def _dpc_clustering(vectors, n_clusters):
    """密度峰值聚类"""
    from sklearn.neighbors import NearestNeighbors

    N = vectors.shape[0]
    # 采样大图像加速
    if N > 5000:
        idx = np.random.choice(N, 5000, replace=False)
        vec_sample = vectors[idx]
    else:
        vec_sample = vectors
        idx = np.arange(N)

    # 近邻距离
    nn = NearestNeighbors(n_neighbors=min(20, len(vec_sample)))
    nn.fit(vec_sample)
    dist, _ = nn.kneighbors(vec_sample)

    # 局部密度 ρ = exp(-mean_dist_to_knn)
    rho = np.exp(-dist[:, 1:].mean(axis=1))

    # δ = min distance to higher-density point
    order = np.argsort(-rho)
    delta = np.zeros_like(rho)
    delta[order[0]] = dist.max()
    for i in range(1, len(order)):
        higher = order[:i]
        delta[order[i]] = np.min(
            np.linalg.norm(vec_sample[order[i]] - vec_sample[higher], axis=1)
        )

    # γ = ρ * δ 选聚类中心
    gamma = rho * delta
    if n_clusters is None:
        # 自动：γ 高于均值+2σ 的点为中心
        threshold = gamma.mean() + 2 * gamma.std()
        centers = np.where(gamma > threshold)[0]
        n_clusters = max(2, len(centers))
        if n_clusters > 10:
            # 取 γ 最高的前 10 个
            centers = np.argsort(-gamma)[:10]
            n_clusters = 10

    # 如果有采样，用 K-means 对全量数据分类
    from sklearn.cluster import KMeans
    prototypes = vec_sample[np.argsort(-gamma)[:n_clusters]]
    km = KMeans(n_clusters=n_clusters, init=prototypes,
                random_state=42, n_init=1)
    labels = km.fit_predict(vectors)
    prototypes = km.cluster_centers_

    scores = _cluster_scores(vectors, labels, prototypes)
    scores['gamma'] = gamma
    scores['rho'] = rho
    scores['delta'] = delta
    return labels, prototypes, scores


def _cluster_scores(vectors, labels, prototypes):
    """聚类质量指标"""
    K = len(prototypes)
    scores = {'n_clusters': K}

    # 簇内紧密度（平均到质心距离）
    intra = []
    cluster_sizes = []
    for k in range(K):
        mask = labels == k
        cluster_sizes.append(mask.sum())
        if mask.sum() > 0:
            d = np.linalg.norm(vectors[mask] - prototypes[k], axis=1).mean()
            intra.append(d)
    scores['intra_cluster_dist'] = np.mean(intra) if intra else 0

    # 簇间分离度（质心间最小距离）
    if K > 1:
        inter = []
        for i in range(K):
            for j in range(i+1, K):
                inter.append(np.linalg.norm(prototypes[i] - prototypes[j]))
        scores['min_inter_cluster_dist'] = np.min(inter)
        scores['davies_bouldin'] = _davies_bouldin(vectors, labels, prototypes)
    else:
        scores['min_inter_cluster_dist'] = 0
        scores['davies_bouldin'] = 0

    scores['cluster_sizes'] = cluster_sizes
    return scores


def _davies_bouldin(vectors, labels, prototypes):
    """Davies-Bouldin 指数（越小越好）"""
    K = len(prototypes)
    if K < 2:
        return 0

    # 每个簇的平均分散度
    scatter = np.zeros(K)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            scatter[k] = np.linalg.norm(
                vectors[mask] - prototypes[k], axis=1
            ).mean()

    db = 0
    for i in range(K):
        max_ratio = 0
        for j in range(K):
            if i != j:
                d_ij = np.linalg.norm(prototypes[i] - prototypes[j])
                if d_ij > 0:
                    ratio = (scatter[i] + scatter[j]) / d_ij
                    max_ratio = max(max_ratio, ratio)
        db += max_ratio
    return db / K


def match_primitives(prototypes_a, prototypes_b):
    """
    跨图基元匹配（Hungarian 算法）。

    匹配两张图的基元，基于余弦相似度。

    Returns:
        matches: dict {idx_a: idx_b} 匹配关系
        similarities: 匹配对的相似度
        unmatched_a, unmatched_b: 未匹配的基元
    """
    Ka, Kb = len(prototypes_a), len(prototypes_b)

    # 归一化
    pa = prototypes_a / (np.linalg.norm(prototypes_a, axis=1, keepdims=True) + 1e-8)
    pb = prototypes_b / (np.linalg.norm(prototypes_b, axis=1, keepdims=True) + 1e-8)

    # 余弦相似度矩阵 [Ka, Kb]
    sim = np.dot(pa, pb.T)

    # Hungarian 最大化总相似度
    cost = 1.0 - sim  # 最小化 1-sim = 最大化 sim
    row_idx, col_idx = linear_sum_assignment(cost)

    matches = {}
    similarities = {}
    for r, c in zip(row_idx, col_idx):
        if sim[r, c] > 0.5:  # 相似度阈值
            matches[int(r)] = int(c)
            similarities[int(r)] = float(sim[r, c])

    unmatched_a = [i for i in range(Ka) if i not in matches]
    unmatched_b = [j for j in range(Kb) if j not in matches.values()]

    return matches, similarities, unmatched_a, unmatched_b


def primitive_to_operator_profile(prototype_vector, operator_dim=8):
    """
    将 64 维基元向量解释为算子贡献。

    64 维 = 8×8 外积展开。每个 8 维块对应一个算子与其他算子的交互。
    取对角线（块内第 i 个元素）= 算子 i 的自交互 = 算子 i 的强度²。

    Returns:
        op_strengths: [8] 每个算子的平均贡献强度
    """
    proto = prototype_vector.reshape(operator_dim, operator_dim)  # [8, 8]
    # 对角线 = 算子自交互
    diag = np.abs(np.diag(proto))
    # 归一化
    diag = diag / (diag.sum() + 1e-8)
    return diag
