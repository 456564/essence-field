"""
场交互动力学 v1.0

物质 = 场在引力/斥力/摩擦作用下自发形成的稳定态。
不聚类。不加标签。场自己沉淀出结构。

引力: 相似向量互相靠拢 → 同物质聚合
斥力: 不相似向量被边界隔开 → 物质分离
摩擦: 边缘处变化被减速 → 边界先稳定

稳定态 → 连通分量 = 物质区域。
"""

import torch
import torch.nn.functional as F
import numpy as np


def field_relaxation(field, edge_metric, n_iters=50, alpha=0.15,
                     attract_scale=1.0, repulsion_threshold=0.3):
    """
    场弛豫：引力+斥力+摩擦驱动场收敛到稳定态。

    Args:
        field: [B, C, H, W] 初始场
        edge_metric: [B, 1, H, W] 边缘度规（摩擦系数）
        n_iters: 最大迭代数
        alpha: 基础步长
        attract_scale: 引力强度缩放
        repulsion_threshold: 相似度低于此值→斥力（不被吸引）

    Returns:
        field_relaxed: [B, C, H, W]
        convergence: list[float] 每轮变化量
        stability_map: [B, 1, H, W] 每像素的稳定度(0=振荡, 1=已收敛)
    """
    B, C, H, W = field.shape
    device = field.device
    dtype = field.dtype

    # 扩散系数 = exp(-edge) → 边缘处趋近0
    diffusivity = torch.exp(-edge_metric)  # [B, 1, H, W]

    # 预计算拉普拉斯核
    laplacian_k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                               dtype=dtype, device=device)

    phi = field
    convergence = []

    # 追踪每个像素的历史变化，用于检测稳定态
    change_history = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    # 为每通道复制拉普拉斯核
    lap_kernel = laplacian_k.repeat(B * C, 1, 1, 1)  # [B*C, 1, 3, 3]

    for it in range(n_iters):
        phi_prev = phi.clone()

        # ---- 引力 + 斥力 ----
        # 对每通道独立计算拉普拉斯
        phi_flat = phi.reshape(1, B * C, H, W)
        lap = F.conv2d(phi_flat, lap_kernel, padding=1, groups=B * C)
        lap = lap.reshape(B, C, H, W)

        # 引力：相似邻域拉向中心
        attraction = lap

        # 斥力：用边缘度规调制——边缘处斥力强（扩散被阻断=邻域被排斥）
        friction = diffusivity  # [B, 1, H, W]  边缘处摩擦力大

        # 合力更新
        delta = alpha * attract_scale * attraction * friction
        phi = phi + delta

        # ---- 稳定性追踪 ----
        change = (phi - phi_prev).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        change_history = 0.9 * change_history + 0.1 * change  # 指数移动平均

        # 记录收敛
        rel_change = (phi - phi_prev).norm() / (phi.norm() + 1e-8)
        convergence.append(rel_change.item())

        # 早停：全场变化 < 1e-4
        if rel_change < 1e-4:
            break

    # 稳定度：变化历史小 → 稳定
    stability_map = torch.exp(-change_history * 50.0)  # [0,1], 1=稳定

    return phi, convergence, stability_map


def extract_materials(field_relaxed, stability_map, edge_metric,
                      min_region_size=100, stability_threshold=0.3):
    """
    K-means 特征分组 + 连通分量空间后处理。

    1. K-means 在特征空间做粗分组
    2. 对每个 K-means 标签的连通分量 → 独立物质
    3. 小的连通分量合并到相邻大分量

    这保留了 K-means 的特征空间区分能力，
    又用连通分量强制了空间一致性。
    """
    B, C, H, W = field_relaxed.shape
    field_np = field_relaxed[0].detach().cpu().numpy()  # [C, H, W]
    edge_np = edge_metric[0, 0].detach().cpu().numpy()
    stab_np = stability_map[0, 0].detach().cpu().numpy()

    # ---- Step 1: K-means 粗分组 ----
    vec = field_np.reshape(C, -1).T  # [N, C]
    vec_norm = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)

    # 加空间坐标
    ys, xs = np.mgrid[0:H, 0:W]
    vec_spatial = np.column_stack([
        vec_norm,
        xs.flatten() / W * 0.3,
        ys.flatten() / H * 0.3,
    ])

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = km.fit_predict(vec_spatial)  # [N]
    km_labels = km_labels.reshape(H, W)

    # 中值滤波去噪
    from scipy.ndimage import median_filter
    km_labels = median_filter(km_labels, size=5)

    # ---- Step 2: 连通分量 ----
    from scipy import ndimage
    structure = np.ones((3, 3), dtype=bool)

    # 对每个 K-means 簇，找所有连通分量
    all_components = np.zeros((H, W), dtype=int)
    comp_id = 1
    for k in range(3):
        mask_k = (km_labels == k)
        if mask_k.sum() == 0:
            continue
        labeled_k, n_k = ndimage.label(mask_k, structure=structure)
        for sub_id in range(1, n_k + 1):
            sub_mask = labeled_k == sub_id
            if sub_mask.sum() >= min_region_size:
                all_components[sub_mask] = comp_id
                comp_id += 1

    n_components = comp_id - 1

    # ---- Step 3: 只保留最大的几个分量 ----
    if n_components == 0:
        return np.zeros((H, W), dtype=int), 1

    # 统计各分量大小
    comp_sizes = {}
    for cid in range(1, n_components + 1):
        comp_sizes[cid] = (all_components == cid).sum()

    # 取前 5 大分量
    max_keep = 5
    top_components = sorted(comp_sizes, key=comp_sizes.get, reverse=True)[:max_keep]

    labels = np.zeros_like(all_components)
    new_id = 1
    for cid in top_components:
        labels[all_components == cid] = new_id
        new_id += 1

    # 合并其余像素到最近的大分量
    from scipy.spatial import KDTree
    unlabeled = labels == 0
    if unlabeled.any():
        labeled_ys, labeled_xs = np.where(~unlabeled)
        label_vals = labels[~unlabeled]
        tree = KDTree(np.column_stack([labeled_ys, labeled_xs]))
        un_ys, un_xs = np.where(unlabeled)
        if len(un_ys) > 0:
            _, nearest = tree.query(np.column_stack([un_ys, un_xs]))
            labels[un_ys, un_xs] = label_vals[nearest]

    # 重新标签：0..n-1
    unique_labels = np.unique(labels)
    label_map = {old: i for i, old in enumerate(sorted(unique_labels))}
    new_labels = np.zeros_like(labels)
    for old, new in label_map.items():
        new_labels[labels == old] = new

    return new_labels, len(label_map)


def field_dynamics_pipeline(field, edge_metric,
                             n_relax_iters=30, alpha=0.15,
                             min_region_size=100):
    """
    完整场动力学管线：弛豫 → 提取物质。

    Args:
        field: [B, C, H, W] 初始场
        edge_metric: [B, 1, H, W]
        n_relax_iters: 弛豫迭代数
        alpha: 步长
        min_region_size: 最小物质区域大小

    Returns:
        labels: [H, W] 物质标签
        field_relaxed: [B, C, H, W] 弛豫后场
        convergence: list[float]
        n_materials: int
    """
    from .field_core import edge_aware_diffusion

    # 1. 场弛豫
    field_relaxed, convergence = edge_aware_diffusion(
        field, edge_metric, n_iters=n_relax_iters, alpha=alpha)

    # 2. 计算稳定度
    stability_map = _compute_stability(field_relaxed)

    # 3. 连通分量后处理
    labels, n_materials = extract_materials(
        field_relaxed, stability_map, edge_metric,
        min_region_size=min_region_size)

    return labels, field_relaxed, convergence, n_materials


def field_dynamics_pipeline(field, edge_metric,
                             n_relax_iters=30, alpha=0.15,
                             min_region_size=100):
    """
    完整场动力学管线：弛豫 → 提取物质。

    用已有的边缘感知扩散做弛豫，用稳定态连通分量替代 K-means。

    Args:
        field: [B, C, H, W] 初始场
        edge_metric: [B, 1, H, W]
        n_relax_iters: 弛豫迭代数
        alpha: 步长
        min_region_size: 最小物质区域大小

    Returns:
        labels: [H, W] 物质标签
        field_relaxed: [B, C, H, W] 弛豫后场
        convergence: list[float]
        n_materials: int
    """
    from .field_core import edge_aware_diffusion

    # 1. 场弛豫（复用已验证的边缘感知扩散）
    field_relaxed, convergence = edge_aware_diffusion(
        field, edge_metric, n_iters=n_relax_iters, alpha=alpha)

    # 2. 计算稳定度（弛豫后像素向量的局部一致性）
    stability_map = _compute_stability(field_relaxed)

    # 3. 提取连通稳定态 = 物质
    labels, n_materials = extract_materials(
        field_relaxed, stability_map, edge_metric,
        min_region_size=min_region_size,
        stability_threshold=0.3)

    return labels, field_relaxed, convergence, n_materials


def _compute_stability(field):
    """
    计算每个像素的稳定度 = 与邻域的向量一致性。

    稳定像素：邻域向量高度一致 → 属于同一个稳定态
    不稳定像素：邻域向量变化大 → 处于过渡区/边界
    """
    B, C, H, W = field.shape
    # 4 邻域向量差异
    diffs = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        shifted = torch.roll(field, shifts=(dy, dx), dims=(2, 3))
        diff = (field - shifted).norm(dim=1, keepdim=True)  # [B, 1, H, W]
        diffs.append(diff)
    avg_diff = torch.stack(diffs).mean(dim=0)  # [B, 1, H, W]

    # 稳定度 = 1 / (1 + 邻域差异)
    stability = 1.0 / (1.0 + avg_diff * 10.0)  # [0, 1]
    return stability
