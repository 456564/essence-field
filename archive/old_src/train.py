"""
训练和评估脚本

支持:
  1. 开环 baseline (直接 ConvNeXt classifier)
  2. 闭环 (ClosedLoopVision)
  3. 对比实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .backbone import BaseVision
from .closed_loop import ClosedLoopVision


class BaselineModel(nn.Module):
    """开环 baseline: ConvNeXt 原装分类头"""
    def __init__(self):
        super().__init__()
        self.backbone = BaseVision(pretrained=True, freeze_backbone=True)
        self.classifier = nn.Linear(768, 1000)

    def forward(self, x):
        feat = self.backbone(x)               # [B, 768, 7, 7]
        feat = feat.mean([-2, -1])             # GAP → [B, 768]
        return self.classifier(feat)           # [B, 1000]


def get_imagenet_val_loader(data_root, batch_size=128, num_workers=4):
    """ImageNet val 数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(
        root=f"{data_root}/val",
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating"):
    """
    计算 top-1 和 top-5 准确率
    """
    model.eval()
    model.to(device)

    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)

        # top-1
        _, pred = logits.topk(1, 1, True, True)
        top1_correct += pred.eq(labels.view(-1, 1)).sum().item()

        # top-5
        _, pred5 = logits.topk(5, 1, True, True)
        top5_correct += pred5.eq(labels.view(-1, 1)).sum().item()

        total += labels.size(0)

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total

    return top1, top5


def compare_baseline_vs_closed_loop(data_root="/data/imagenet",
                                     batch_size=128,
                                     device="cuda"):
    """
    对比实验：开环 vs 闭环
    """
    print("=" * 60)
    print("对比实验: Baseline (开环) vs Closed-Loop (闭环)")
    print("=" * 60)

    # 数据
    loader = get_imagenet_val_loader(data_root, batch_size)

    # Baseline
    print("\n[1/2] 评估 Baseline (开环 ConvNeXt) ...")
    baseline = BaselineModel()
    top1_base, top5_base = evaluate(baseline, loader, device, desc="Baseline")

    # Closed Loop
    print("\n[2/2] 评估 Closed-Loop (闭环) ...")
    closed_loop = ClosedLoopVision(num_iters=2, freeze_backbone=True)
    top1_cl, top5_cl = evaluate(closed_loop, loader, device, desc="Closed-Loop")

    # 结果
    print("\n" + "=" * 60)
    print(f"{'模型':<25} {'Top-1':<10} {'Top-5':<10}")
    print("-" * 45)
    print(f"{'Baseline (开环)':<25} {top1_base:<10.2f} {top5_base:<10.2f}")
    print(f"{'Closed-Loop (闭环)':<25} {top1_cl:<10.2f} {top5_cl:<10.2f}")
    print(f"{'差距':<25} {top1_cl - top1_base:<+10.2f} {top5_cl - top5_base:<+10.2f}")
    print("=" * 60)

    return {
        "baseline_top1": top1_base,
        "baseline_top5": top5_base,
        "closed_loop_top1": top1_cl,
        "closed_loop_top5": top5_cl,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/data/imagenet")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    compare_baseline_vs_closed_loop(
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=args.device,
    )
