"""
方案 B：不改 ConvNeXt 一行代码，外面套预测循环

流程：
  ConvNeXt (原封不动) → 特征 [B, 768]
                           ↓
  预测循环（迭代 N 轮）：
    状态 → 预测预期特征
    预期 vs 实际 → 误差 → 更新状态
                           ↓
  最终状态 → 分类器
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class PredictionLayer(nn.Module):
    """
    预测层：挂在 ConvNeXt 外面

    输入：状态向量 [B, 256]
    输出：预期特征 [B, 768] + 增益 [B, 1]
    """
    def __init__(self, state_dim=256, feat_dim=768):
        super().__init__()

        # 预测器：状态 → 预期特征
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

        # 误差 → 状态更新
        self.updater = nn.Linear(feat_dim, state_dim)

        # 增益调节器
        self.gain = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, actual_feat):
        # 1. 从状态预测"应该看到什么特征"
        expected = self.predictor(state)

        # 2. 误差（stop_grad 避免预测器坍缩）
        error = actual_feat - expected.detach()

        # 3. 增益
        g = self.gain(state)

        # 4. 更新状态
        state = state + g * self.updater(error)

        return state, expected, error


class ClosedLoopOnFeatures(nn.Module):
    """
    方案 B：ConvNeXt 不改一行代码
    外面套一个预测循环
    """
    def __init__(self, num_iters=3, state_dim=256, num_classes=1000):
        super().__init__()

        # ConvNeXt 原封不动，冻结
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        for p in self.features.parameters():
            p.requires_grad = False

        # 预测层（新加的，可训练）
        self.predictor = PredictionLayer(state_dim, 768)

        # 可学习的初始状态
        self.init_state = nn.Parameter(torch.zeros(1, state_dim))

        # 分类头（新加的，可训练）
        self.classifier = nn.Linear(state_dim, num_classes)

        self.num_iters = num_iters

    def forward(self, x, return_all=False):
        # 1. ConvNeXt 提特征（一次，不改动）
        with torch.no_grad():
            feat = self.features(x)                 # [B, 768, 7, 7]
            feat = feat.mean([-2, -1])              # GAP → [B, 768]

        # 2. 初始化状态
        state = self.init_state.expand(x.size(0), -1)

        # 3. 预测循环
        states = [state]
        errors = []
        expecteds = []

        for i in range(self.num_iters):
            state, expected, error = self.predictor(state, feat)
            states.append(state)
            errors.append(error.norm())
            expecteds.append(expected)

        # 4. 分类
        logits = self.classifier(states[-1])

        if return_all:
            return {
                "logits": logits,
                "states": states,
                "errors": errors,
                "expecteds": expecteds,
                "features": feat,
            }

        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ClosedLoopOnFeatures(num_iters=3).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params/1e6:.1f}M")
    print(f"可训练参数量: {trainable_params/1e6:.1f}M")

    # 随机输入，跑一次 forward
    x = torch.randn(1, 3, 224, 224).to(device)
    out = model(x, return_all=True)

    print(f"\n=== 前向测试 ===")
    print(f"输入: {x.shape}")
    print(f"输出 logits: {out['logits'].shape}")
    print(f"预测类别: {out['logits'].argmax(dim=1).item()}")
    print(f"迭代轮数: {len(out['states'])}")
    for i, (s, e) in enumerate(zip(out['states'], out['errors'])):
        print(f"  第{i}轮 state范数={s.norm().item():.4f}  误差={e.item():.4f}")

    print(f"\n✅ 跑通了，代码没问题")
