"""
八卦可视化工具

提取每个子网络的空间激活图，观察"先天偏向"。
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt

from .subnets import BAGUA_REGISTRY


def compute_activation_maps(net, img_tensor):
    """
    提取子网络最后一个 conv 层的空间激活图
    返回: [H, W] 热力图 (0~1)
    """
    features = {}

    def hook_fn(module, input, output):
        features['feat'] = output.detach()

    last_conv = None
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m

    if last_conv is None:
        return None

    handle = last_conv.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = net(img_tensor)
    handle.remove()

    if 'feat' not in features:
        return None

    feat = features['feat']  # [B, C, H, W]
    act = feat[0].abs().mean(dim=0).cpu().numpy()

    # 缩放到 224×224
    act = cv2.resize(act, (224, 224))

    # 归一化
    mn, mx = act.min(), act.max()
    if mx > mn:
        act = (act - mn) / (mx - mn)
    return act


def visualize_all(img_tensor, img_disp, output_path, device="cpu"):
    """绘制 8 卦的空间激活图"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    # 英文标签（避免 matplotlib 中文缺失）
    EN_NAMES = ["Heaven\nQian", "Earth\nKun", "Thunder\nZhen", "Wind\nXun",
                "Water\nKan", "Fire\nLi", "Mountain\nGen", "Lake\nDui"]
    for idx, (name, cn, desc, NetClass) in enumerate(BAGUA_REGISTRY):
        net = NetClass().to(device).eval()
        act = compute_activation_maps(net, img_tensor)

        ax = axes[idx]
        ax.imshow(img_disp, alpha=0.6)
        if act is not None:
            ax.imshow(act, cmap='hot', alpha=0.4, vmin=0, vmax=1)
        ax.set_title(f"{EN_NAMES[idx]}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path
