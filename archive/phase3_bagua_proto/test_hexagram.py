"""
八卦→64卦 最小实现

8 个子网络各自提取图像特征
两两配对（外积）→ 64 维卦象表示
看不同输入是否能产生不同的 64 维模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt


# ─── 8 卦子网络 ─────────────────────────────────────────

BAGUA_NAMES = ["Qian", "Dui", "Li", "Zhen", "Xun", "Kan", "Gen", "Kun"]
#             乾(天)  兑(泽)  离(火)  震(雷)  巽(风)  坎(水)  艮(山)  坤(地)


class BaguaSubNet(nn.Module):
    """
    一个卦象子网络：
    输入图像 → 输出一个固定长度的特征向量
    每个子网络结构相同，初始化不同
    """
    def __init__(self, seed, out_dim=8):
        super().__init__()
        torch.manual_seed(seed)
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, out_dim),
            nn.Tanh(),  # 输出在 [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)  # [B, out_dim]


class BaguaToHexagram(nn.Module):
    """
    8 卦 → 64 卦
    
    输入：8 个特征向量 [B, 8] × 8
    运算：两两外积
    输出：64 维卦象向量
    """
    def __init__(self, feat_dim=8):
        super().__init__()
        self.feat_dim = feat_dim
    
    def forward(self, features):
        """
        features: list of 8 tensors [B, feat_dim]
        returns: [B, 64] hexagram
        """
        B = features[0].size(0)
        
        # 堆叠成 [B, 8, feat_dim]
        stacked = torch.stack(features, dim=1)  # [B, 8, feat_dim]
        
        # 外积：每对卦象之间的交互
        # [B, 8, feat_dim] × [B, feat_dim, 8] → [B, 8, 8]
        hexagram = torch.bmm(
            stacked,                    # [B, 8, feat_dim]
            stacked.transpose(1, 2)     # [B, feat_dim, 8]
        )
        
        # 展平 → [B, 64]
        hexagram = hexagram.view(B, -1)  # [B, 64]
        
        return hexagram, stacked


# ─── 完整模型 ─────────────────────────────────────────────

class HexagramModel(nn.Module):
    """
    八卦→64卦 视觉模型
    
    8 子网络 → 外积 → 64 维卦象 → 分类头
    """
    def __init__(self, feat_dim=8, num_classes=5):
        super().__init__()
        
        # 8 卦子网络（不同种子）
        seeds = [42, 123, 256, 512, 1024, 2048, 4096, 8192]
        self.subnets = nn.ModuleList([
            BaguaSubNet(seed=seeds[i], out_dim=feat_dim)
            for i in range(8)
        ])
        
        # 64 卦 → 分类
        self.hex_to_class = nn.Linear(64, num_classes)
        
        # 外积层
        self.outer = BaguaToHexagram(feat_dim)
    
    def forward(self, x, return_all=False):
        # 8 个子网络并行
        features = [subnet(x) for subnet in self.subnets]
        
        # 外积 → 64 卦
        hexagram, stacked = self.outer(features)
        
        # 分类
        logits = self.hex_to_class(hexagram)
        
        if return_all:
            return logits, hexagram, features
        
        return logits


# ─── 测试 ─────────────────────────────────────────────────

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 模型
    model = HexagramModel(feat_dim=8, num_classes=5).to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数总量: {n_params/1e3:.0f}K")
    print(f"  8 子网络: {sum(p.numel() for p in model.subnets.parameters())/1e3:.0f}K")
    print(f"  64→分类: {sum(p.numel() for p in model.hex_to_class.parameters())/1e3:.0f}K")
    
    # 加载图片
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    print(f"\n输入: {img_tensor.shape}")
    
    # 前向
    with torch.no_grad():
        logits, hexagram, features = model(img_tensor, return_all=True)
    
    hex_64 = hexagram[0].cpu().numpy()
    
    # ─── 分析 ───
    print(f"\n{'='*60}")
    print("8 卦子网络输出")
    print(f"{'='*60}")
    
    for i, (name, feat) in enumerate(zip(BAGUA_NAMES, features)):
        vals = feat[0].cpu().numpy()
        info = f"  {name}: [{', '.join(f'{v:.3f}' for v in vals)}]"
        print(info)
    
    # 子网络之间的差异
    print(f"\n子网络两两差异 (cosine 距离):")
    for i in range(8):
        for j in range(i+1, 8):
            vi = features[i][0].cpu().numpy()
            vj = features[j][0].cpu().numpy()
            cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8)
            print(f"  {BAGUA_NAMES[i]}-{BAGUA_NAMES[j]}: {1-cos_sim:.4f}")
    
    # 64 卦分析
    print(f"\n{'='*60}")
    print("64 卦分析")
    print(f"{'='*60}")
    print(f"  卦象向量: {hex_64.shape}")
    print(f"  范围: {hex_64.min():.4f} ~ {hex_64.max():.4f}")
    print(f"  前 16 个值: {[f'{v:.3f}' for v in hex_64[:16]]}")
    
    # 64 卦矩阵可视化（画出 8×8 热力图）
    hex_mat = hex_64.reshape(8, 8)
    
    # 保存图表
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子网络特征（8 卦条形图）
    ax = axes[0]
    feat_matrix = np.array([f[0].cpu().numpy() for f in features])  # [8, 8]
    ax.imshow(feat_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([f'F{i}' for i in range(8)])
    ax.set_yticklabels(BAGUA_NAMES)
    ax.set_title("8 Trigrams (卦) × 8 features")
    
    # 64 卦热力图
    ax = axes[1]
    im = ax.imshow(hex_mat, cmap='viridis')
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(BAGUA_NAMES)
    ax.set_yticklabels(BAGUA_NAMES)
    ax.set_title("64 Hexagrams (64卦) — outer product")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    out_path = str(output_dir / f"hexagram_{Path(args.image).stem}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  结果图: {out_path}")
    
    print(f"\n{'='*60}")
    print("当前是随机权重，8 个卦象只是在随机初始化下不同")
    print("需要训练后，才会学到有意义的卦象维度")
    print(f"{'='*60}")


if __name__ == "__main__":
    test()
