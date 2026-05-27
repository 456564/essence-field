"""
CIFAR-100 训练（预提特征版）

先把所有图过一遍 ConvNeXt 存特征，再用特征训练预测层。
ConvNeXt 冻结不改，所以特征只需算一次，训练极快。

用法：
  python train_cifar.py                    # 完整 30 epoch
  python train_cifar.py --epochs 5         # 快速验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time


# ─── 设备 ─────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ─── 预提特征 ─────────────────────────────────────────────

@torch.no_grad()
def extract_features(loader, model):
    """把整批数据过 ConvNeXt，只存特征 [N, 768]"""
    all_feats, all_labels = [], []
    for images, labels in tqdm(loader, desc="Extracting features"):
        feats = model(images.to(device))          # [B, 768, 7, 7]
        feats = feats.mean([-2, -1])               # GAP → [B, 768]
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def prepare_data(batch_size=128):
    """准备数据集并预提特征"""
    # CIFAR-100
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_set = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    val_set = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    # 加载 ConvNeXt 特征提取器（冻结）
    backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    features = backbone.features

    # 预提特征
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print("预提训练集特征...")
    train_feats, train_labels = extract_features(train_loader, features)
    print(f"  训练特征: {train_feats.shape}")

    print("预提验证集特征...")
    val_feats, val_labels = extract_features(val_loader, features)
    print(f"  验证特征: {val_feats.shape}")

    # 转成 TensorDataset 训练时加载
    train_data = TensorDataset(train_feats, train_labels)
    val_data = TensorDataset(val_feats, val_labels)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

    return train_loader, val_loader


# ─── 模型定义 ─────────────────────────────────────────────

class Baseline(nn.Module):
    """开环：特征 → 分类头"""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 100),
        )

    def forward(self, x):
        return self.classifier(x)


class PredictionLayer(nn.Module):
    def __init__(self, state_dim=256, feat_dim=768):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.updater = nn.Linear(feat_dim, state_dim)
        self.gain = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, actual_feat):
        expected = self.predictor(state)
        error = actual_feat - expected.detach()
        g = self.gain(state)
        state = state + g * self.updater(error)
        return state, expected, error


class ClosedLoop(nn.Module):
    """闭环：特征 → 预测循环 → 分类"""
    def __init__(self, num_iters=3, state_dim=256):
        super().__init__()
        self.predictor = PredictionLayer(state_dim, 768)
        self.init_state = nn.Parameter(torch.zeros(1, state_dim))
        self.classifier = nn.Linear(state_dim, 100)
        self.num_iters = num_iters

    def forward(self, feat, return_all=False):
        B = feat.size(0)
        state = self.init_state.expand(B, -1)
        states, errors = [], []

        for _ in range(self.num_iters):
            state, expected, error = self.predictor(state, feat)
            states.append(state)
            errors.append(error.norm(dim=1).mean().item())

        logits = self.classifier(states[-1])

        if return_all:
            return logits, states, errors
        return logits

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── 训练 ─────────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs=30, lr=1e-3, name="Model"):
    model.to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = logits.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                _, pred = logits.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        val_acc = 100.0 * correct / total
        best_acc = max(best_acc, val_acc)

        t = time.time() - t0
        print(f"  {epoch+1:2d}/{epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"train={train_acc:.1f}%  val={val_acc:.1f}%  "
              f"best={best_acc:.1f}%  {t:.0f}s/epoch")

        scheduler.step()

    return best_acc


# ─── 主流程 ───────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    # 1. 准备数据（含预提特征）
    train_loader, val_loader = prepare_data()

    # 2. Baseline
    print(f"\n{'='*50}")
    print("Baseline（开环：线性分类头）")
    baseline = Baseline()
    print(f"可训练参数: {sum(p.numel() for p in baseline.parameters() if p.requires_grad)/1e3:.0f}K")
    acc_baseline = train(baseline, train_loader, val_loader, args.epochs, args.lr, "Baseline")

    # 3. Closed-Loop
    print(f"\n{'='*50}")
    print(f"Closed-Loop（预测层 × {args.iters} 轮）")
    cl = ClosedLoop(num_iters=args.iters)
    print(f"可训练参数: {sum(p.numel() for p in cl.parameters() if p.requires_grad)/1e3:.0f}K")
    acc_cl = train(cl, train_loader, val_loader, args.epochs, args.lr * 0.5, "Closed-Loop")

    # 4. 结果
    print(f"\n{'='*50}")
    print("最终对比")
    print(f"  Baseline (开环):     {acc_baseline:.1f}%")
    print(f"  Closed-Loop (闭环):  {acc_cl:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
