"""
Caltech-101 5 类训练 + 遮挡测试

验证闭环模型在"物体被挡住一半"时能不能比开环更好地识别。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ─── 5 类物体 ─────────────────────────────────────────────

CLASS_NAMES = ["cup", "lamp", "chair", "panda", "camera"]


# ─── 数据 ─────────────────────────────────────────────────

def load_data(root="./data", train_ratio=0.7):
    """加载 Caltech-101，过滤出 5 类，切分训练/测试"""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # 灰度图 → RGB
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full = datasets.Caltech101(root=root, download=True, transform=transform)

    # 找出目标类对应的索引
    target_ids = []
    for name in CLASS_NAMES:
        for i, cat in enumerate(full.categories):
            if cat == name:
                target_ids.append(i)
                break

    # 过滤样本
    all_indices = []
    all_labels_new = []
    for idx, (_, label) in enumerate(full):
        if label in target_ids:
            all_indices.append(idx)
            all_labels_new.append(target_ids.index(label))

    # 7:3 切分
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.3, stratify=all_labels_new, random_state=42
    )

    train_set = Subset(full, train_idx)
    test_set = Subset(full, test_idx)

    print(f"训练: {len(train_set)} 张, 测试: {len(test_set)} 张")
    return train_set, test_set


@torch.no_grad()
def precompute_features(loader, feature_extractor, keep_4d=True):
    """
    预提特征。
    keep_4d=True  → 保留 [B, 768, 7, 7]（需要做遮挡测试）
    keep_4d=False → GAP 后 [B, 768]（更省内存）
    """
    all_feats, all_labels = [], []
    for images, labels in tqdm(loader, desc="Features"):
        feats = feature_extractor(images.to(device))  # [B, 768, 7, 7]
        if not keep_4d:
            feats = feats.mean([-2, -1])              # [B, 768]
        all_feats.append(feats.cpu())
        all_labels.append(labels)

    return torch.cat(all_feats), torch.cat(all_labels)


# ─── 模型 ─────────────────────────────────────────────────

class Baseline(nn.Module):
    """开环：特征 → 分类头"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        # x 可以是 [B, 768]（已 GAP）或 [B, 768, 7, 7]（未 GAP）
        if x.dim() == 4:
            x = x.mean([-2, -1])
        return self.classifier(x)


class Predictor(nn.Module):
    def __init__(self, state_dim=256, feat_dim=768):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Linear(state_dim, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.update = nn.Linear(feat_dim, state_dim)
        self.gain = nn.Sequential(nn.Linear(state_dim, 1), nn.Sigmoid())

    def forward(self, state, feat):
        expected = self.predict(state)
        error = feat - expected.detach()
        g = self.gain(state)
        new_state = state + g * self.update(error)
        return new_state, expected, error


class ClosedLoop(nn.Module):
    def __init__(self, num_iters=3, state_dim=256, num_classes=5):
        super().__init__()
        self.predictor = Predictor(state_dim, 768)
        self.init_state = nn.Parameter(torch.zeros(1, state_dim))
        self.classifier = nn.Linear(state_dim, num_classes)
        self.num_iters = num_iters

    def forward(self, feat):
        if feat.dim() == 4:
            feat = feat.mean([-2, -1])
        B = feat.size(0)
        state = self.init_state.expand(B, -1)
        for _ in range(self.num_iters):
            state, _, _ = self.predictor(state, feat)
        return self.classifier(state)


# ─── 训练 ─────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, name=""):
    model.to(device)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best = 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, corr, tot = 0, 0, 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(feats), labels)
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            _, pred = model(feats).max(1)
            corr += pred.eq(labels).sum().item()
            tot += labels.size(0)

        train_acc = 100.0 * corr / tot

        # eval
        model.eval()
        corr, tot = 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                _, pred = model(feats).max(1)
                corr += pred.eq(labels).sum().item()
                tot += labels.size(0)

        val_acc = 100.0 * corr / tot
        best = max(best, val_acc)
        t = time.time() - t0
        print(f"  {ep+1:2d}/{epochs}  loss={loss_sum/len(train_loader):.3f}  "
              f"train={train_acc:.1f}  val={val_acc:.1f}  best={best:.1f}  {t:.0f}s")

        optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs).step()

    return best


# ─── 遮挡测试 ─────────────────────────────────────────────

def make_occlusion(feats_4d, ratio, position="bottom"):
    """
    feats_4d: [B, 768, 7, 7]  特征图
    返回: [B, 768] GAP 后的遮挡特征
    """
    B, C, H, W = feats_4d.shape
    mask = torch.ones(B, 1, H, W, device=feats_4d.device)

    h = int(H * ratio)
    if position == "bottom":
        mask[:, :, -h:, :] = 0
    elif position == "center":
        h0 = (H - h) // 2
        mask[:, :, h0:h0+h, h0:h0+h] = 0

    return (feats_4d * mask).mean([-2, -1])


@torch.no_grad()
def occlusion_benchmark(model, feats_4d, labels, ratios=[0, 0.25, 0.5, 0.75]):
    """不同遮挡比例下的准确率"""
    model.eval()
    model.to(device)
    feats_4d, labels = feats_4d.to(device), labels.to(device)

    results = {}
    for r in ratios:
        if r == 0:
            inp = feats_4d.mean([-2, -1])
        else:
            inp = make_occlusion(feats_4d, r)
        _, pred = model(inp).max(1)
        results[r] = (pred == labels).float().mean().item() * 100

    return results


# ─── 主流程 ───────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # 1. 加载数据
    print(f"{'='*50}")
    print(f"Caltech-101 子集: {CLASS_NAMES}")
    train_set, test_set = load_data()

    # 2. 预提特征（保留 4D 用于遮挡测试）
    backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    backbone.eval()
    fe = backbone.features

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    print("\n预提训练特征...")
    train_feats, train_labels = precompute_features(train_loader, fe, keep_4d=False)
    print(f"  训练: {train_feats.shape}")

    print("预提测试特征（保留 4D 用于遮挡测试）...")
    test_feats_4d, test_labels = precompute_features(test_loader, fe, keep_4d=True)
    print(f"  测试: {test_feats_4d.shape}")

    train_data = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # 把测试集也转成 [B, 768] 给训练用
    test_feats_pooled = test_feats_4d.mean([-2, -1])
    test_data = TensorDataset(test_feats_pooled, test_labels)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 3. Baseline
    print(f"\n{'='*50}")
    print("Baseline（开环：特征 → 线性分类头）")
    base = Baseline(num_classes=5)
    n = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"可训练参数: {n/1e3:.0f}K")
    acc_base = train_model(base, train_loader, test_loader, args.epochs, args.lr, "Baseline")

    # 4. Closed-Loop
    print(f"\n{'='*50}")
    print(f"Closed-Loop（预测层 × {args.iters} 轮）")
    cl = ClosedLoop(num_iters=args.iters, num_classes=5)
    n = sum(p.numel() for p in cl.parameters() if p.requires_grad)
    print(f"可训练参数: {n/1e3:.0f}K")
    acc_cl = train_model(cl, train_loader, test_loader, args.epochs, args.lr * 0.5, "Closed-Loop")

    # 5. 遮挡对比
    print(f"\n{'='*50}")
    print("遮挡测试（底部遮挡）")
    print(f"{'遮挡':<8} {'Baseline':<10} {'Closed-Loop':<12} {'差值':<10}")
    print("-" * 40)

    r_base = occlusion_benchmark(base, test_feats_4d, test_labels)
    r_cl = occlusion_benchmark(cl, test_feats_4d, test_labels)

    for r in [0, 0.25, 0.5, 0.75]:
        d = r_cl[r] - r_base[r]
        print(f"{r*100:<8.0f}% {r_base[r]:<10.1f} {r_cl[r]:<12.1f} {d:<+10.1f}")

    print(f"\n{'='*50}")
    print("完成!")


if __name__ == "__main__":
    main()
