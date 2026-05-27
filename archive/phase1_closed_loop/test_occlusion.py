"""
遮挡测试：开环 vs 闭环

用实物照片，遮挡部分后，对比识别结果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from pathlib import Path

from test_closed_loop import ClosedLoopOnFeatures


def load_image(path):
    """加载图片，预处理到 224×224"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, 224, 224]


def occlude_image(tensor, ratio=0.5, position="bottom"):
    """
    遮挡图像的一部分
    """
    t = tensor.clone()
    H, W = 224, 224
    device = tensor.device
    mask = torch.ones(1, 1, H, W, device=device)

    h = int(H * ratio)
    w = int(W * ratio)

    if position == "bottom":
        mask[:, :, -h:, :] = 0
    elif position == "top":
        mask[:, :, :h, :] = 0
    elif position == "left":
        mask[:, :, :, :w] = 0
    elif position == "right":
        mask[:, :, :, -w:] = 0
    elif position == "center":
        h0, w0 = (H - h) // 2, (W - w) // 2
        mask[:, :, h0:h0+h, w0:w0+w] = 0

    # 用 mean 填充遮挡区域
    mean_val = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    t = t * mask + mean_val * (1 - mask)
    return t, mask


def get_imagenet_labels():
    """加载 ImageNet 1000 类别名称"""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    import urllib.request
    try:
        with urllib.request.urlopen(url) as f:
            return [line.strip() for line in f.readlines()]
    except:
        return [str(i) for i in range(1000)]


@torch.no_grad()
def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--position", type=str, default="bottom",
                        choices=["bottom", "top", "left", "right", "center"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载 ConvNeXt 完整的预训练模型
    full_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    full_model.eval()

    # 打印分类头结构，看看实际长什么样
    print(f"分类头类型: {type(full_model.classifier)}")
    if hasattr(full_model.classifier, 'in_features'):
        print(f"  输入维度: {full_model.classifier.in_features}")
    if isinstance(full_model.classifier, torch.nn.Sequential):
        for i, layer in enumerate(full_model.classifier):
            print(f"  第{i}层: {layer}")

    # 用全模型作为 baseline（端到端）
    baseline_features = full_model.features

    # 加载图片
    x = load_image(args.image).to(device)
    x_occ, _ = occlude_image(x, ratio=args.ratio, position=args.position)

    # 闭环模型
    closed_loop = ClosedLoopOnFeatures(num_iters=3).to(device)
    closed_loop.eval()

    labels = get_imagenet_labels()

    print(f"\n{'='*60}")
    print(f"图片: {args.image}")
    print(f"遮挡: {args.position} {args.ratio*100:.0f}%")
    print(f"{'='*60}")

    # ─── 测试 1: Baseline 原图 ───
    logits = full_model(x)  # 整体 forward
    probs = F.softmax(logits, dim=1)
    top5 = probs.topk(5)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).item()

    print(f"\n--- Baseline (原图, 预训练权重) ---")
    print(f"  不确定性(熵): {entropy:.4f}")
    for i in range(5):
        idx = top5.indices[0, i].item()
        pct = top5.values[0, i].item() * 100
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  {pct:.1f}% → [{idx}] {label}")

    # ─── 测试 2: Baseline 遮挡图 ───
    logits_occ = full_model(x_occ)  # 整体 forward
    probs_occ = F.softmax(logits_occ, dim=1)
    top5_occ = probs_occ.topk(5)
    entropy_occ = -(probs_occ * torch.log(probs_occ + 1e-8)).sum(dim=1).item()

    print(f"\n--- Baseline (遮挡图, 预训练权重) ---")
    print(f"  不确定性(熵): {entropy_occ:.4f}")
    for i in range(5):
        idx = top5_occ.indices[0, i].item()
        pct = top5_occ.values[0, i].item() * 100
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  {pct:.1f}% → [{idx}] {label}")

    # ─── 测试 3: 闭环 遮挡图 ───
    out = closed_loop(x_occ, return_all=True)
    logits_cl = out["logits"]
    probs_cl = F.softmax(logits_cl, dim=1)
    top5_cl = probs_cl.topk(5)
    entropy_cl = -(probs_cl * torch.log(probs_cl + 1e-8)).sum(dim=1).item()

    print(f"\n--- Closed-Loop (遮挡图, 随机权重) ---")
    print(f"  不确定性(熵): {entropy_cl:.4f}")
    for i in range(5):
        idx = top5_cl.indices[0, i].item()
        pct = top5_cl.values[0, i].item() * 100
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  {pct:.1f}% → [{idx}] {label}")

    # ─── 测试 4: 闭环各轮次状态变化（看预测循环是否在工作） ───
    print(f"\n--- 闭环迭代调试 ---")
    for i, (s, e) in enumerate(zip(out["states"], out["errors"])):
        print(f"  第{i}轮 state范数={s.norm().item():.4f}  误差={e.item():.4f}")

    print(f"\n{'='*60}")
    print("结论: baseline 原图应该能正确认出马克杯")
    print("      baseline 遮挡后置信度应该大跌")
    print("      闭环目前权重随机，输出仅供参考")


if __name__ == "__main__":
    test()
