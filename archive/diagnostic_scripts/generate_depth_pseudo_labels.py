"""
为 ImageNet 生成深度伪标签

使用 Depth Anything 模型（MiDaS 团队）为每张 ImageNet 图生成深度图。
一次运行，产出的伪标签可以被物理预测层用作训练信号。

用法:
  python scripts/generate_depth_pseudo_labels.py \
      --data-root /path/to/imagenet \
      --output-dir /path/to/depth_labels
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse


def load_depth_model(device):
    """
    加载 Depth Anything 或 MiDaS v3.1 深度估计模型
    以 MiDaS 为例（Depth Anything 需要额外安装）
    """
    try:
        # 尝试 Depth Anything
        from depth_anything.dpt import DepthAnything
        model = DepthAnything()
        model.load_state_dict(
            torch.load("checkpoints/depth_anything_vitl14.pth"))
    except ImportError:
        # 回退到 MiDaS
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval()

    model.to(device)
    return model


@torch.no_grad()
def generate_labels(model, loader, output_dir, device):
    """
    对 DataLoader 中每张图生成深度图并保存为 .npy
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    for images, labels in tqdm(loader, desc="Generating depth"):
        filenames = loader.dataset.samples  # 需要适配实际路径

        depth_maps = model(images.to(device))
        depth_maps = nn.functional.interpolate(
            depth_maps.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        for depth, path in zip(depth_maps.cpu(), filenames):
            save_path = output_dir / f"{Path(path[0]).stem}.npy"
            np.save(save_path, depth.numpy().astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./depth_labels")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 加载模型
    print(f"Loading depth model on {args.device}...")
    model = load_depth_model(args.device)

    # 数据集
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=f"{args.data_root}/val", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 生成
    generate_labels(model, loader, args.output_dir, args.device)
    print(f"Depth pseudo-labels saved to {args.output_dir}")
