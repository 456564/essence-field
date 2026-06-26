"""
本质场核心 — 零可学习参数版本

核心假设:
  8 算子 = 物质几何的完整描述。不需要可学习投影来"抽象"。
  场 = 算子在"边缘感知度规"下的平滑解。
  物质 = 平滑后场的连通分量。

管线:
  算子响应 → 稳健归一化 → 空间预平滑 → 边缘感知扩散 → 聚类 → 物质

零可学习参数。零黑箱。
"""

import torch
import torch.nn.functional as F
import numpy as np


def robust_normalize(tensor, lo_pct=2, hi_pct=98):
    """
    稳健归一化到 [0, 1]。
    用百分位而非 min/max，抗离群点。
    """
    B, C, H, W = tensor.shape
    flat = tensor.view(B, -1)
    n = flat.shape[1]
    lo = flat.kthvalue(max(1, int(lo_pct / 100 * n)), dim=1,
                       keepdim=True).values.view(B, C, 1, 1)
    hi = flat.kthvalue(min(n, int(hi_pct / 100 * n)), dim=1,
                       keepdim=True).values.view(B, C, 1, 1)
    spread = hi - lo
    # 处理常数张量
    spread = torch.where(spread < 1e-8, torch.ones_like(spread), spread)
    return torch.clamp((tensor - lo) / spread, 0, 1)


def spatial_presmooth(tensor, sigma=3.0):
    """
    空间预平滑：高斯模糊。
    目的：消除像素级噪声，让同物质内部像素的算子响应趋于一致。
    sigma 控制平滑程度——越大越粗粒度。
    """
    ks = int(sigma * 4) + 1
    if ks % 2 == 0:
        ks += 1
    # 构造高斯核
    xs = torch.arange(ks, device=tensor.device).float() - ks // 2
    gauss_1d = torch.exp(-0.5 * (xs / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d.view(1, -1) * gauss_1d.view(-1, 1)
    kernel = kernel.view(1, 1, ks, ks).to(tensor.dtype)
    # 对每通道独立卷积
    B, C, H, W = tensor.shape
    out = F.conv2d(tensor.view(-1, 1, H, W), kernel, padding=ks // 2,
                   groups=1)
    return out.view(B, C, H, W)


def compute_edge_metric(operator_maps, edge_scale=5.0, close_radius=2):
    """
    从算子计算边缘感知度规。
    度规 d(p, q) = |p-q|_spatial * (1 + edge * edge_scale)

    边缘 = 震(拉普拉斯) + 离(梯度幅值) 的组合。
    度规值高 → 场跨越此处困难 → 物质边界自然形成。

    闭运算（先膨胀后腐蚀）：填充薄缝（如键帽间隙），
    只保留厚轮廓（如物体边界）。

    Args:
        close_radius: 形态学闭运算半径。越大填充越宽的缝隙。

    Returns:
        metric: [B, 1, H, W] 度规强度（0 = 无边缘，大 = 强边缘）
    """
    zhen = operator_maps['zhen']
    li = operator_maps['li']
    edge = robust_normalize(zhen) + robust_normalize(li)
    edge = robust_normalize(edge)  # [0, 1]

    # 形态学闭运算：填充薄缝，保留厚边
    if close_radius > 0:
        B, C, H, W = edge.shape
        ks = 2 * close_radius + 1
        # 膨胀（max pooling）
        dilated = F.max_pool2d(
            F.pad(edge, [close_radius]*4, mode='reflect'),
            ks, 1, 0)
        # 腐蚀（-max_pool on negative）
        eroded = -F.max_pool2d(
            F.pad(-dilated, [close_radius]*4, mode='reflect'),
            ks, 1, 0)
        edge = eroded

    return edge * edge_scale


def appearance_features(x):
    """
    表象层 — 局部相对特征（像素 vs 邻域的差异）。

    不用绝对颜色——两个灰色物体应有不同的几何签名，
    而不是因为"都是灰色"而相似。

    Returns:
        appearance: [B, 6, H, W]
          ch0: 亮度局部对比度 ( |L - L_local_mean| )
          ch1: 色相局部对比度 ( |A - A_local_mean| + |B - B_local_mean| )
          ch2: 纹理粗糙度 (5x5 方差 / 15x15 方差 → 细 vs 粗纹理)
          ch3: 局部边缘密度 (9x9 窗口 Sobel 均值)
          ch4: 局部方向一致性 (Sobel 梯度方向方差倒数)
          ch5: 高光强度 (亮度 > 局部均值+2σ)
    """
    B, C, H, W = x.shape

    # 亮度
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    L = 0.299 * r + 0.587 * g + 0.114 * b

    # 色相近似
    A_ch = (r - g) * 0.5 + 0.5
    B_ch = (b - (r + g) * 0.5) * 0.5 + 0.5

    # box filters
    box5 = torch.ones(1, 1, 5, 5, device=x.device) / 25
    box15 = torch.ones(1, 1, 15, 15, device=x.device) / 225
    box9 = torch.ones(1, 1, 9, 9, device=x.device) / 81

    # ch0: 亮度局部对比度（相对邻域均值的绝对偏差）
    L_local = F.conv2d(L, box5, padding=2)
    L_contrast = torch.abs(L - L_local)

    # ch1: 色相局部对比度
    A_local = F.conv2d(A_ch, box5, padding=2)
    B_local = F.conv2d(B_ch, box5, padding=2)
    chroma_contrast = torch.abs(A_ch - A_local) + torch.abs(B_ch - B_local)

    # ch2: 纹理粗糙度（细纹理 5x5变/15x15变 接近1，粗纹理 远小于1）
    L_mean5 = F.conv2d(L, box5, padding=2)
    L_var5 = F.conv2d(L * L, box5, padding=2) - L_mean5 * L_mean5
    L_mean15 = F.conv2d(L, box15, padding=7)
    L_var15 = F.conv2d(L * L, box15, padding=7) - L_mean15 * L_mean15
    texture_scale = (L_var5 + 1e-8) / (L_var15 + 1e-8)

    # ch3: 局部边缘密度
    sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],
                           dtype=x.dtype, device=x.device)
    sobel_y = sobel_x.transpose(2, 3)
    L_grad = torch.sqrt(
        F.conv2d(L, sobel_x, padding=1)**2 +
        F.conv2d(L, sobel_y, padding=1)**2 + 1e-8
    )
    edge_density = F.conv2d(L_grad, box9, padding=4)

    # ch4: 局部方向一致性（梯度方向在邻域内的集中度）
    gx = F.conv2d(L, sobel_x, padding=1)
    gy = F.conv2d(L, sobel_y, padding=1)
    angle = torch.atan2(gy, gx + 1e-8)
    # 方向一致性 = 邻域内角度方差倒数
    angle_mean = F.conv2d(angle, box9, padding=4)
    angle_var = F.conv2d((angle - angle_mean)**2, box9, padding=4) + 1e-8
    direction_consistency = 1.0 / (angle_var * 10 + 1.0)

    # ch5: 高光强度（亮度远超邻域均值的程度）
    L_local_std = torch.sqrt((F.conv2d(L*L, box9, padding=4) -
                               F.conv2d(L, box9, padding=4)**2).clamp(min=1e-8))
    highlight = ((L - F.conv2d(L, box9, padding=4)) / (L_local_std + 1e-8)).clamp(min=0)

    # 堆叠 + 归一化
    chs = [L_contrast, chroma_contrast, texture_scale,
           edge_density, direction_consistency, highlight]
    normed = [robust_normalize(ch) for ch in chs]
    appearance = torch.cat(normed, dim=1)  # [B, 6, H, W]

    return appearance


def essence_field_compute(x, pipe, presmooth_sigma=3.0, edge_scale=5.0,
                          use_appearance=True):
    """
    三层本质场 — 表象 + 抽象 → 本质。

    表象层: 颜色纹理亮度 [B, 6, H, W]
    抽象层: 8 几何算子 [B, 8, H, W]
    本质层: 拼接后扩散 → 14 维联合场

    Returns:
        field: [B, D, H, W] 本质场 (D=14 或 D=8)
        edge_metric: [B, 1, H, W]
        operator_maps: dict
    """
    from .operators import BASE_OPS

    with torch.no_grad():
        raw_ops = pipe.operator_layer.base_ops(x)

    # --- 抽象层：8 几何算子 ---
    norm_ops = {}
    for name in BASE_OPS:
        norm_ops[name] = robust_normalize(raw_ops[name])
    smooth_ops = {}
    for name in BASE_OPS:
        smooth_ops[name] = spatial_presmooth(norm_ops[name], presmooth_sigma)
    op_vecs = [smooth_ops[name] for name in BASE_OPS]
    abstract_field = torch.cat(op_vecs, dim=1)  # [B, 8, H, W]

    # --- 表象层 ---
    if use_appearance:
        appearance_field = appearance_features(x)  # [B, 6, H, W]
        # 对表象也做空间预平滑（降噪）
        appearance_field = spatial_presmooth(appearance_field, presmooth_sigma)
        # 拼接：表象在前 6 维，抽象在后 8 维
        field = torch.cat([appearance_field, abstract_field], dim=1)  # [B, 14, H, W]
    else:
        field = abstract_field  # [B, 8, H, W]

    # 边缘度规（抽象层算子计算，不受表象影响）
    edge_metric = compute_edge_metric(norm_ops, edge_scale, close_radius=2)

    return field, edge_metric, norm_ops


def edge_aware_diffusion(field, edge_metric, n_iters=20, alpha=0.2):
    """
    边缘感知各向异性扩散。

    核心物理：每个像素的向量向邻域扩散，
    但扩散系数受边缘度规调制——边缘处扩散被阻断。

    迭代求解：Φ_{t+1} = Φ_t + α * div(g(edge) * ∇Φ_t)
    其中 g(edge) = exp(-edge_metric)

    稳定的数值实现：用加权局部平均近似散度算子。

    Args:
        field: [B, C, H, W] 待扩散的场
        edge_metric: [B, 1, H, W] 边缘度规
        n_iters: 迭代次数
        alpha: 扩散步长

    Returns:
        field_smoothed: [B, C, H, W]
        convergence: list[float]
    """
    B, C, H, W = field.shape
    # 扩散系数
    diffusivity = torch.exp(-edge_metric)  # [B, 1, H, W], 边缘处→0

    # 预计算拉普拉斯核（4邻域加权）
    # 用卷积实现：对每个方向分别计算
    phi = field
    convergence = []

    for it in range(n_iters):
        # 4邻域拉普拉斯：∇·(g∇Φ) ≈ Σ_{dir} g_i * (Φ_neighbor - Φ_center)
        laplacian = torch.zeros_like(phi)

        # 水平方向
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = torch.roll(phi, shifts=(dy, dx), dims=(2, 3))
            g_shifted = torch.roll(diffusivity, shifts=(dy, dx), dims=(2, 3))
            g_avg = (diffusivity + g_shifted) / 2.0  # 调和平均
            laplacian += g_avg * (shifted - phi)

        # 更新
        phi_new = phi + alpha * laplacian

        delta = (phi_new - phi).norm() / (phi.norm() + 1e-8)
        convergence.append(delta.item())
        phi = phi_new

    return phi, convergence


def field_to_materials(field_8, edge_metric=None, n_clusters=None):
    """
    从本质场提取物质。

    先边缘感知扩散，再聚类。

    Args:
        field_8: [B, D, H, W] 本质向量场（D=8 或 D=14）
        edge_metric: [B, 1, H, W] 边缘度规（None = 用均匀度规）
        n_clusters: 聚类数（None = 自动）

    Returns:
        labels: [H, W] 物质标签
        field_smoothed: [B, 8, H, W] 扩散后场
        convergence: list[float]
    """
    # 扩散
    if edge_metric is None:
        edge_metric = torch.zeros(1, 1, field_8.shape[2], field_8.shape[3],
                                  device=field_8.device)
    field_smooth, convergence = edge_aware_diffusion(field_8, edge_metric)

    # 转 numpy 做聚类
    C = field_smooth.shape[1]
    vec = field_smooth[0].permute(1, 2, 0).reshape(-1, C).detach().cpu().numpy()
    H, W = field_8.shape[2], field_8.shape[3]
    N = H * W

    # L2 归一化算子向量
    vec_norm = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)

    # 添加空间坐标 → 空间邻近也影响聚类归属
    ys, xs = np.mgrid[0:H, 0:W]
    ys_norm = ys.flatten() / H  # [0, 1]
    xs_norm = xs.flatten() / W  # [0, 1]
    # 空间权重：相邻像素空间距离小 → 更容易归同簇
    spatial_weight = 0.3  # 空间 vs 算子的权重比（越大越重视空间邻近）
    vec_spatial = np.column_stack([
        vec_norm,
        xs_norm * spatial_weight,
        ys_norm * spatial_weight,
    ])  # [N, 10]

    from sklearn.cluster import KMeans
    if n_clusters is None:
        inertias = []
        for k in range(2, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(vec_spatial)
            inertias.append(km.inertia_)
        curves = [inertias[i-1] - 2*inertias[i] + inertias[i+1]
                  for i in range(1, len(inertias)-1)]
        best_k = 2 + np.argmax(np.abs(curves))
    else:
        best_k = n_clusters

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(vec_spatial)
    labels = labels.reshape(H, W)

    return labels, field_smooth, convergence, km.cluster_centers_[:, :8]


def describe_material(labels, field_8, operator_names=None):
    """
    描述每种物质的操作子特征。

    Args:
        labels: [H, W] 物质标签
        field_8: [B, 8, H, W] 本质场（扩散后）

    Returns:
        materials: list[dict]
    """
    C = field_8.shape[1]
    if operator_names is None:
        if C == 14:
            operator_names = [
                '亮对比', '色相对比', '纹理粗细', '边缘密度', '方向一致', '高光',
                '乾qian(圆)', '坤kun(容器)', '震zhen(边)', '巽xun(纹)',
                '坎kan(曲)', '离li(能)', '艮gen(块)', '兑dui(比)'
            ]
        else:
            operator_names = [
                '乾qian(圆)', '坤kun(容器)', '震zhen(边)', '巽xun(纹)',
                '坎kan(曲)', '离li(能)', '艮gen(块)', '兑dui(比)'
            ]

    vec = field_8[0].permute(1, 2, 0).reshape(-1, C).detach().cpu().numpy()
    H, W = labels.shape
    K = labels.max() + 1

    materials = []
    for k in range(K):
        mask = labels == k
        area = mask.sum() / (H * W)
        if mask.sum() > 0:
            prototype = vec[mask.flatten()].mean(axis=0)
        else:
            prototype = np.zeros(8)
        top3 = np.argsort(-prototype)[:3]
        top_str = ', '.join(f'{operator_names[j]}={prototype[j]:.3f}'
                           for j in top3)
        materials.append({
            'mask': mask,
            'area': area,
            'prototype': prototype,
            'top_operators': top_str,
        })

    materials.sort(key=lambda m: m['area'], reverse=True)
    return materials
