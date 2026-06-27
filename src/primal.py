"""
Primal Field v1 — minimum primal, single rule, multi-faceted primal vector

Philosophy:
  pixel's nature = neighbors influence each other, strength = similarity
  primal vector = RGB(color face) + neighbor diffs(structure face) + local var(texture face)
                = multi-faceted — same pixel expresses on multiple faces simultaneously
"""

import torch
import torch.nn.functional as F
import numpy as np


def enrich_field(rgb_field):
    """
    RGB(3) -> RGB + structure = 8-dim primal vector.

    ch0-2: R,G,B (color face)
    ch3-6: up/down/left/right neighbor diff (structure face)
    ch7:   5x5 local variance (texture face)

    Structure not named — naming is human's job after emergence.
    """
    B, C, H, W = rgb_field.shape
    device = rgb_field.device

    # Color face
    rgb = rgb_field

    # Structure face: 4-direction neighbor diffs
    diffs = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nb = torch.roll(rgb_field, shifts=(dy, dx), dims=(2, 3))
        diff = (rgb_field - nb).abs().mean(dim=1, keepdim=True)
        diffs.append(diff)
    structure = torch.cat(diffs, dim=1)  # [B, 4, H, W]

    # Texture face: 5x5 local variance (avg over RGB channels)
    kernel = torch.ones(1, 1, 5, 5, device=device) / 25
    gray = rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    local_mean = F.conv2d(gray, kernel, padding=2)
    local_sq_mean = F.conv2d(gray * gray, kernel, padding=2)
    local_var = (local_sq_mean - local_mean * local_mean).clamp(min=0)

    # Concat: color(3) + structure(4) + texture(1) = 8-dim
    return torch.cat([rgb, structure, local_var], dim=1)


def primal_relax(field, tau=0.05, alpha=0.3, n_iters=50, repulsion=0.0):
    """Pure attraction (+ optional repulsion). Rule 1+2."""
    B, C, H, W = field.shape
    phi = field.clone()
    conv = []

    for _ in range(n_iters):
        prev = phi.clone()
        attract = torch.zeros_like(phi)
        awsum = torch.zeros(B, 1, H, W, device=field.device)
        repel = torch.zeros_like(phi)
        rwsum = torch.zeros(B, 1, H, W, device=field.device)

        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = torch.roll(phi, shifts=(dy, dx), dims=(2, 3))
            diff = (nb - phi).pow(2).sum(dim=1, keepdim=True)
            sim = torch.exp(-diff / tau)
            attract += nb * sim
            awsum += sim
            if repulsion > 0:
                push = (phi - nb) * (1.0 - sim)
                repel += push
                rwsum += (1.0 - sim)

        phi_attract = attract / (awsum + 1e-8)
        if repulsion > 0:
            phi_repel = phi + repel / (rwsum + 1e-8)
            phi_new = phi_attract * (1 - repulsion) + phi_repel * repulsion
        else:
            phi_new = phi_attract

        phi = (1 - alpha) * phi + alpha * phi_new
        d = (phi - prev).norm() / (phi.norm() + 1e-8)
        conv.append(d.item())
        if d < 1e-4:
            break

    return phi, conv


def primal_relax_full(field, tau=0.05, alpha=0.3, n_iters=50,
                       use_repulsion=False, repulsion_strength=0.1,
                       use_multiscale=False, coarse_weight=0.15,
                       use_inertia=False, inertia_gain=0.3):
    """All 4 rules switchable. Rule 1+2+3+4."""
    if not use_repulsion and not use_multiscale and not use_inertia:
        return primal_relax(field, tau, alpha, n_iters)

    B, C, H, W = field.shape
    phi = field.clone()
    conv = []
    stability = torch.zeros(B, 1, H, W, device=field.device) if use_inertia else None

    for it in range(n_iters):
        prev = phi.clone()

        # Attraction + Repulsion
        attract = torch.zeros_like(phi)
        awsum = torch.zeros(B, 1, H, W, device=field.device)
        repel = torch.zeros_like(phi)
        rwsum = torch.zeros(B, 1, H, W, device=field.device)

        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = torch.roll(phi, shifts=(dy, dx), dims=(2, 3))
            diff = (nb - phi).pow(2).sum(dim=1, keepdim=True)
            sim = torch.exp(-diff / tau)
            attract += nb * sim
            awsum += sim
            if use_repulsion:
                push = (phi - nb) * (1.0 - sim)
                repel += push
                rwsum += (1.0 - sim)

        phi_attract = attract / (awsum + 1e-8)
        if use_repulsion:
            phi_repel = phi + repel / (rwsum + 1e-8)
            phi_new = phi_attract * (1 - repulsion_strength) + phi_repel * repulsion_strength
        else:
            phi_new = phi_attract

        # Multi-scale
        if use_multiscale:
            coarse = F.interpolate(phi, scale_factor=0.5, mode='bilinear')
            coarse, _ = primal_relax(coarse, tau=tau * 2.0, alpha=alpha * 0.3, n_iters=1)
            coarse_up = F.interpolate(coarse, size=(H, W), mode='bilinear')
            phi_new = phi_new * (1 - coarse_weight) + coarse_up * coarse_weight

        # Inertia (consecutive stability counter)
        if use_inertia:
            change = (phi_new - phi).abs().mean(dim=1, keepdim=True)
            is_stable = change < 0.001
            stability = torch.where(is_stable, stability + 1, torch.zeros_like(stability))
            adaptive_alpha = alpha / (1.0 + stability * inertia_gain)
        else:
            adaptive_alpha = alpha

        phi = (1 - adaptive_alpha) * phi + adaptive_alpha * phi_new
        d = (phi - prev).norm() / (phi.norm() + 1e-8)
        conv.append(d.item())
        if d < 1e-4:
            break

    return phi, conv


def extract_domains(field_relaxed, grad_pct=75, min_domain_size=None):
    """Extract material domains from relaxed field via field gradient."""
    B, C, H, W = field_relaxed.shape
    field_np = field_relaxed[0].detach().cpu().numpy()

    gy = np.abs(np.diff(field_np, axis=1, append=field_np[:, -1:, :])).mean(0)
    gx = np.abs(np.diff(field_np, axis=2, append=field_np[:, :, -1:])).mean(0)
    grad = gy + gx
    interior = grad < np.percentile(grad, grad_pct)

    from scipy import ndimage
    labels, n_raw = ndimage.label(interior)

    if min_domain_size is None:
        min_domain_size = max(50, H * W // 100)

    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n_raw + 1))
    new_labels = np.zeros_like(labels)
    next_id = 1
    for lid in range(1, n_raw + 1):
        if sizes[lid - 1] >= min_domain_size:
            new_labels[labels == lid] = next_id
            next_id += 1

    labels = new_labels - 1
    labels = labels.clip(min=0)
    n_domains = labels.max() + 1 if labels.max() >= 0 else 0
    return labels, n_domains


def hierarchical_decompose(field, tau=0.02, alpha=0.3, n_iters=50,
                            flags=8, max_depth=3, min_size=200):
    """
    层级递归——在每个稳定域内部再次弛豫，直到不再分裂。

    流水遇石则分，遇潭则止。
    场遇到内部结构则递归，遇到均匀区域则停止。

    Args:
        field: [B, C, H, W] primal vector
        tau, alpha, n_iters: primal parameters
        flags: rule combination
        max_depth: maximum recursion depth
        min_size: minimum pixel count to recurse into

    Returns:
        hierarchy: list of dicts [{label, depth, mask, parent}]
    """
    hierarchy = []
    _recurse(field, tau, alpha, n_iters, flags, 0, max_depth,
             min_size, None, hierarchy)
    return hierarchy


def _recurse(field, tau, alpha, n_iters, flags, depth, max_depth,
             min_size, parent_label, hierarchy):
    """递归核心——收敛检查：子域无明显分化→停止"""
    if depth > max_depth:
        return

    labels, n_domains = extract_domains(field)

    # 收敛: 不分 → 停止
    if n_domains <= 1:
        return
    # 最大域占 >80% → 碎片 → 停止
    sizes = [(labels == k).sum() for k in range(n_domains)]
    sizes.sort(reverse=True)
    if sizes[0] > sum(sizes) * 0.8:
        return
    # 子域不足 3 个 → 无递归意义
    if n_domains < 2:
        return

    B, C, H, W = field.shape
    field_np = field[0].cpu().numpy()

    for k in range(n_domains):
        mask = labels == k
        size = mask.sum()
        if size < min_size:
            continue

        label_id = len(hierarchy)
        hierarchy.append({
            'label': label_id, 'depth': depth, 'mask': mask,
            'parent': parent_label, 'size_pct': size / (H * W) * 100,
        })

        # 递归: 裁剪子区域，重新弛豫
        if size > min_size * 3 and depth < max_depth:
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            child_region = field_np[:, y0:y1, x0:x1]
            child_tensor = torch.from_numpy(child_region).unsqueeze(0).to(
                field.device)
            _recurse(child_tensor, tau, alpha, n_iters, flags,
                     depth + 1, max_depth, min_size, label_id, hierarchy)
