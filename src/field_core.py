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


def compute_essence_edge(norm_ops, appearance_field=None, edge_scale=5.0,
                         close_radius=2):
    """
    联合边缘度规 — 表象梯度 + 抽象算子梯度的最大值。

    单zhen只检测强度边缘。联合度规捕获所有边界类型：
      强度边(震/离)、纹理边(艮)、曲率边(坎)、容器边(坤)、颜色边(表象)

    零参数——所有梯度都是Sobel卷积。

    Returns:
        metric: [B, 1, H, W] 联合边缘强度
    """
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                           dtype=list(norm_ops.values())[0].dtype,
                           device=list(norm_ops.values())[0].device)
    sobel_y = sobel_x.transpose(2, 3)

    B = list(norm_ops.values())[0].shape[0]
    H, W = list(norm_ops.values())[0].shape[2:]
    device = list(norm_ops.values())[0].device

    combined = torch.zeros(B, 1, H, W, device=device)

    # 8个抽象算子的梯度
    for name, op in norm_ops.items():
        gx = F.conv2d(op, sobel_x, padding=1)
        gy = F.conv2d(op, sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
        combined = torch.max(combined, grad)

    # 表象层的梯度（如果有）
    if appearance_field is not None:
        for c in range(appearance_field.shape[1]):
            ch = appearance_field[:, c:c+1]
            gx = F.conv2d(ch, sobel_x, padding=1)
            gy = F.conv2d(ch, sobel_y, padding=1)
            grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
            combined = torch.max(combined, grad)

    # 归一化 + 闭运算填充薄缝
    combined = robust_normalize(combined)
    if close_radius > 0:
        ks = 2 * close_radius + 1
        dilated = F.max_pool2d(
            F.pad(combined, [close_radius]*4, mode='reflect'), ks, 1, 0)
        eroded = -F.max_pool2d(
            F.pad(-dilated, [close_radius]*4, mode='reflect'), ks, 1, 0)
        combined = eroded

    return combined * edge_scale


# ═══════════════════════════════════════════════════════════
# 全局上下文特征（零参数）
# ═══════════════════════════════════════════════════════════

def multi_scale_operators(norm_ops, fine_sigma=1.0, coarse_sigma=8.0):
    """
    Step 1: 像素 ↔ 全局（多尺度分桶）。

    每个算子取两个尺度：
      fine:   原图（局部几何细节）
      coarse: 大核模糊（全局结构——"我在全图中的地位"）

    Returns:
        ms_field: [B, 16, H, W] (8 算子 × 2 尺度)
    """
    from .operators import BASE_OPS
    op_names = list(BASE_OPS.keys())
    fine_list = []
    coarse_list = []

    for name in op_names:
        v = norm_ops[name]  # [B, 1, H, W]
        fine_list.append(v)
        # 粗尺度：大核高斯模糊 → 全局上下文
        coarse = spatial_presmooth(v, sigma=coarse_sigma)
        coarse_list.append(coarse)

    ms_field = torch.cat(fine_list + coarse_list, dim=1)  # [B, 16, H, W]
    return ms_field


def operator_wrap_features(norm_ops):
    """
    Step 2: 像素 ↔ 算子间全局关联（环绕度）。

    对每个算子 A，计算"像素 i 被其他算子的梯度环绕的程度"。
    物理含义：高 kun_wrap = 我周围环形区域里，其他算子的梯度都指向外 → 我是被围合的容器内部。

    实现：对算子 A，在像素 i 周围取环形邻域的 B 的梯度，计算平均向心度。

    Returns:
        wrap_field: [B, 8, H, W] 每算子的环绕度
    """
    from .operators import BASE_OPS
    op_names = list(BASE_OPS.keys())

    B = list(norm_ops.values())[0].shape[0]
    H, W = list(norm_ops.values())[0].shape[2:]
    device = list(norm_ops.values())[0].device
    dtype = list(norm_ops.values())[0].dtype

    # Sobel 梯度核
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                           dtype=dtype, device=device)

    wrap_list = []
    for name_A in op_names:
        A = norm_ops[name_A]  # [B, 1, H, W]

        # 其他 7 个算子的梯度平均
        outer_grad = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        for name_B in op_names:
            if name_B == name_A:
                continue
            B_val = norm_ops[name_B]
            gx = F.conv2d(B_val, sobel_x, padding=1)
            gy = F.conv2d(B_val, sobel_x.transpose(2, 3), padding=1)
            outer_grad += torch.sqrt(gx ** 2 + gy ** 2 + 1e-8) / 7.0

        # 在 A 值高的像素周围，外层梯度是否强
        # 用距离加权：环形邻域（半径 5-15）的梯度均值
        r_inner, r_outer = 3, 15
        ks = r_outer * 2 + 1
        box = torch.ones(1, 1, ks, ks, device=device, dtype=dtype) / (ks * ks)
        outer_grad_ring = F.conv2d(
            F.pad(outer_grad, [r_outer]*4, mode='reflect'),
            box)
        # 减去内环（近邻梯度不算"环绕"）
        ks_inner = r_inner * 2 + 1
        box_inner = torch.ones(1, 1, ks_inner, ks_inner, device=device, dtype=dtype) / (ks_inner * ks_inner)
        outer_grad_near = F.conv2d(
            F.pad(outer_grad, [r_inner]*4, mode='reflect'),
            box_inner)
        ring_grad = (outer_grad_ring - outer_grad_near).clamp(min=0)

        # 环绕度 = A 值 × 外环梯度（A 高 + 外环梯度高 = 我被围合）
        wrap = A * ring_grad
        wrap_list.append(wrap)

    wrap_field = torch.cat(wrap_list, dim=1)  # [B, 8, H, W]
    return wrap_field


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
                          use_appearance=True, use_global_context=True):
    """
    本质场计算。

    局部层:   8 几何算子 [B, 8, H, W]
    全局层:   多尺度(16维) + 算子环绕(8维) = [B, 24, H, W]
    表象层:   局部颜色对比度 [B, 6, H, W]  (可选)
    本质层:   拼接后扩散 → [B, D, H, W]

    D = 8 + 24 + 6 = 38 (全部)
      = 8 + 24     = 32 (无表象)
      = 8          = 8  (纯局部)

    Returns:
        field: [B, D, H, W]
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
    abstract_field = torch.cat(op_vecs, dim=1)  # [B, 8, H, W]  抽象层

    fields = [abstract_field]

    # --- 表象层 (6维) ---
    if use_appearance:
        app = appearance_features(x)
        app = spatial_presmooth(app, presmooth_sigma)

        # --- 跨层融合: 表象⊗抽象外积 → 48维 ---
        B, _, H, W = app.shape
        app_exp = app.unsqueeze(2)           # [B, 6, 1, H, W]
        abs_exp = abstract_field.unsqueeze(1) # [B, 1, 8, H, W]
        cross = (app_exp * abs_exp).reshape(B, 48, H, W)
        cross = spatial_presmooth(cross, presmooth_sigma * 0.5)
        fields.append(cross)
        # 也保留原始表象（作为对比基准）
        fields.append(app)

    # --- 全局上下文层 (24维) ---
    if use_global_context:
        # Step 1: 多尺度 (16维 = 8×2)
        ms = multi_scale_operators(norm_ops)
        ms = spatial_presmooth(ms, presmooth_sigma)
        fields.append(ms)

        # Step 2: 算子环绕 (8维)
        wrap = operator_wrap_features(norm_ops)
        wrap = spatial_presmooth(wrap, presmooth_sigma)
        fields.append(wrap)

    field = torch.cat(fields, dim=1)

    # 联合边缘度规 = 表象梯度 + 抽象算子梯度
    appearance_for_edge = appearance_features(x) if use_appearance else None
    edge_metric = compute_essence_edge(norm_ops, appearance_for_edge,
                                        edge_scale, close_radius=2)

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
    从本质场提取物质——稳定态连通分量（替换 K-means）。

    先边缘感知扩散，再找场的"盆地"（稳定态），
    每个盆地 = 一种物质。不需要指定 K。

    Args:
        field_8: [B, D, H, W] 本质向量场
        edge_metric: [B, 1, H, W] 边缘度规
        n_clusters: 忽略（保留兼容性）

    Returns:
        labels: [H, W] 物质标签
        field_smoothed: [B, D, H, W] 扩散后场
        convergence: list[float]
        prototypes: [K, D] 物质原型向量
    """
    # 扩散
    if edge_metric is None:
        edge_metric = torch.zeros(1, 1, field_8.shape[2], field_8.shape[3],
                                  device=field_8.device)
    field_smooth, convergence = edge_aware_diffusion(field_8, edge_metric)

    C = field_smooth.shape[1]
    H, W = field_8.shape[2], field_8.shape[3]
    field_np = field_smooth[0].detach().cpu().numpy()  # [C, H, W]
    edge_np = edge_metric[0, 0].detach().cpu().numpy()

    # ---- Step 1: 计算场的局部梯度（场变化幅度）----
    grad_y = np.abs(np.diff(field_np, axis=1, append=field_np[:, -1:, :]))
    grad_x = np.abs(np.diff(field_np, axis=2, append=field_np[:, :, -1:]))
    field_grad = np.sqrt(np.mean(grad_y**2 + grad_x**2, axis=0))  # [H, W]

    # ---- Step 2: 找盆地中心——场梯度极小 + 远离边缘 ----
    from scipy import ndimage
    from skimage.feature import peak_local_max

    # 盆地高程 = 场梯度 + 边缘度规
    elevation = field_grad + edge_np * 1.5
    elevation = ndimage.gaussian_filter(elevation, sigma=7.0)

    # 找局部极小值 = 物质中心
    minima = peak_local_max(
        -elevation,  # 反转 → 盆地底 = 峰
        min_distance=30,
        threshold_abs=-np.percentile(elevation, 40),
        exclude_border=True,
        num_peaks=6)

    if len(minima) <= 1:
        # 找不到足够盆地 → 回退到单个物质
        labels = np.zeros((H, W), dtype=int)
        prototypes = np.array([field_np.mean(axis=(1, 2))])
        return labels, field_smooth, convergence, prototypes

    # ---- Step 3: 流域分割 ----
    from skimage.segmentation import watershed
    markers = np.zeros((H, W), dtype=int)
    for i, (y, x) in enumerate(minima):
        markers[y, x] = i + 1

    labels = watershed(elevation, markers)

    # ---- Step 4: 合并小区域 → 最近大邻居 ----
    min_size = H * W // 20  # 最小~5%面积
    n_labels = labels.max()
    # 先标记大小
    sizes = {lid: (labels == lid).sum() for lid in range(1, n_labels + 1)}
    big_labels = {lid for lid, sz in sizes.items() if sz >= min_size}

    # 小区域像素重新分配给最近的大区域
    if big_labels:
        from scipy.spatial import KDTree
        # 大区域的坐标
        big_mask = np.isin(labels, list(big_labels))
        big_ys, big_xs = np.where(big_mask)
        big_vals = labels[big_mask]
        tree = KDTree(np.column_stack([big_ys, big_xs]))

        # 小区域坐标
        small_mask = ~big_mask & (labels > 0)
        if small_mask.any():
            small_ys, small_xs = np.where(small_mask)
            _, nearest = tree.query(np.column_stack([small_ys, small_xs]))
            labels[small_ys, small_xs] = big_vals[nearest]

    # 重新标签为 0, 1, 2, ...
    unique = sorted(set(labels[labels > 0]))
    final_labels = np.zeros_like(labels)
    prototypes_list = []
    for new_id, old_id in enumerate(unique, 1):
        mask = labels == old_id
        final_labels[mask] = new_id
        prototypes_list.append(field_np[:, mask].mean(axis=1))

    if not prototypes_list:
        labels = np.zeros((H, W), dtype=int)
        prototypes = np.array([field_np.mean(axis=(1, 2))])
        return labels, field_smooth, convergence, prototypes

    labels = final_labels
    prototypes = np.array(prototypes_list)
    n_materials = len(unique)

    # ---- Step 5: 填充未标记像素 ----
    unlabeled = labels == 0
    if unlabeled.any() and n_materials > 0:
        from scipy.spatial import KDTree
        lys, lxs = np.where(~unlabeled)
        lvals = labels[~unlabeled]
        tree = KDTree(np.column_stack([lys, lxs]))
        uys, uxs = np.where(unlabeled)
        _, nearest = tree.query(np.column_stack([uys, uxs]))
        labels[uys, uxs] = lvals[nearest]

    # 确保 0-indexed
    labels = labels.clip(min=1) - 1

    return labels, field_smooth, convergence, prototypes


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
        if C >= 62:  # 抽象 + 交叉 + 表象 + 全局
            operator_names = (
                ['乾','坤','震','巽','坎','离','艮','兑'] +           # 0-7
                [f'交{i}' for i in range(48)] +                       # 8-55 表象⊗抽象
                ['亮对比','色相对比','纹理粗细','边缘密度','方向一致','高光'] # 56-61
            )
        elif C >= 38:  # 抽象 + 表象 + 全局
            operator_names = [
                '乾','坤','震','巽','坎','离','艮','兑',
                '亮对比','色相对比','纹理粗细','边缘密度','方向一致','高光',
                '乾细','坤细','震细','巽细','坎细','离细','艮细','兑细',
                '乾粗','坤粗','震粗','巽粗','坎粗','离粗','艮粗','兑粗',
                '乾环','坤环','震环','巽环','坎环','离环','艮环','兑环',
            ]
        elif C == 14:
            operator_names = [
                '亮对比', '色相对比', '纹理粗细', '边缘密度', '方向一致', '高光',
                '乾', '坤', '震', '巽', '坎', '离', '艮', '兑'
            ]
        else:
            operator_names = ['乾','坤','震','巽','坎','离','艮','兑']

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
