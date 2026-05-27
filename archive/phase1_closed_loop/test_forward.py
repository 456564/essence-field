"""
最简单的闭环模型 forward 测试
用随机噪声作为输入，验证每层能跑通、形状正确
"""

import torch
from src.closed_loop import ClosedLoopVision

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 创建模型
    model = ClosedLoopVision(
        num_iters=2,
        state_dim=256,
        pretrained_backbone=True,
        trainable_backbone=True,
    ).to(device)
    model.eval()

    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 随机输入模拟一张图
    x = torch.randn(1, 3, 224, 224).to(device)

    # 不带中间结果
    logits = model(x)
    pred = logits.argmax(dim=1)
    print(f"\n=== 基础 forward 测试 ===")
    print(f"输入形状: {x.shape}")
    print(f"输出 logits 形状: {logits.shape}")
    print(f"预测类别: {pred.item()}")

    # 带中间结果
    out = model(x, return_all=True)
    print(f"\n=== 闭环迭代调试 ===")
    print(f"迭代轮数: {len(out['states'])}")
    for i, (s, e) in enumerate(zip(out['states'], out['errors'])):
        print(f"  第{i}轮 state范数={s.norm().item():.4f}, error范数={e.mean().item():.4f}")

    print("\n✅ 前向测试通过！")

if __name__ == "__main__":
    test()
