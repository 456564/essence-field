"""
五行子网络测试

5 个相同结构的小 CNN，不同初始化 → 各自从图像中提取不同维度的信息。
5 个特征图 → 基本单元生成器 → 3D 单元 → 交集 = 物品
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms


# ─── 五行的子网络（结构相同，初始化不同）───

class OneSubNet(nn.Module):
    """一个子网络：输入图像 → 输出特征图"""
    def __init__(self, init_seed):
        super().__init__()
        torch.manual_seed(init_seed)
        
        # 小编码器
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 112
        x = self.pool(F.relu(self.conv2(x)))   # 56
        x = self.pool(F.relu(self.conv3(x)))   # 28
        return x  # [B, 64, 28, 28]


class FusionToUnits(nn.Module):
    """融合层：5 个特征图 → 3D 基本单元"""
    def __init__(self):
        super().__init__()
        # 5 个通道的特征 → 预测 3D 单元参数
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 5, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 1),  # 4 通道: x_offset, y_offset, z, p
        )
    
    def forward(self, feature_maps):
        """
        feature_maps: list of 5 [B, 64, 28, 28]
        
        每个空间位置预测一个基本单元：
          (x_offset, y_offset) — 相对于网格的位置偏移
          z — 深度
          p — 存在概率 (sigmoid)
        """
        # 拼接 5 个特征图
        combined = torch.cat(feature_maps, dim=1)  # [B, 320, 28, 28]
        
        # 预测单元参数
        unit_params = self.fusion(combined)  # [B, 4, 28, 28]
        
        # 解析
        x_offset = torch.tanh(unit_params[:, 0:1]) * 0.5
        y_offset = torch.tanh(unit_params[:, 1:2]) * 0.5
        z = torch.tanh(unit_params[:, 2:3])
        p = torch.sigmoid(unit_params[:, 3:4])
        
        # 生成网格坐标
        B, _, H, W = unit_params.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=unit_params.device),
            torch.linspace(-1, 1, W, device=unit_params.device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        
        # 最终位置 = 网格坐标 + 偏移
        x = grid_x + x_offset
        y = grid_y + y_offset
        
        # 基本单元 = (x, y, z, p)
        units = torch.cat([x, y, z, p], dim=1)  # [B, 4, 28, 28]
        
        return units


class FiveElementsModel(nn.Module):
    """五行模型：5 子网络 + 融合"""
    def __init__(self):
        super().__init__()
        # 5 个子网络，用不同种子初始化
        self.subnets = nn.ModuleList([
            OneSubNet(seed) for seed in [42, 123, 256, 512, 1024]
        ])
        self.fusion = FusionToUnits()
    
    def forward(self, x):
        # 5 个特征图
        features = [net(x) for net in self.subnets]
        
        # 差异度量：5 个特征图有多不同
        diffs = []
        for i in range(5):
            for j in range(i+1, 5):
                diff = F.mse_loss(features[i], features[j])
                diffs.append(diff.item())
        
        # 融合 → 基本单元
        units = self.fusion(features)
        
        return units, features, diffs


# ─── 测试 ─────────────────────────────────────────────────

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 加载模型
    model = FiveElementsModel().to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params/1e3:.0f}K")
    
    # 加载图片
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])(img_rgb).unsqueeze(0).to(device)
    
    print(f"输入: {img_tensor.shape}")
    
    # 前向
    with torch.no_grad():
        units, features, diffs = model(img_tensor)
    
    # ─── 分析 ───
    print(f"\n{'='*60}")
    print("5 个子网络输出的差异 (MSE)")
    print(f"{'='*60}")
    names = ["Metal", "Wood", "Water", "Fire", "Earth"]
    idx = 0
    for i in range(5):
        for j in range(i+1, 5):
            print(f"  {names[i]} vs {names[j]}: {diffs[idx]:.4f}")
            idx += 1
    
    # 综合差异度（值越大 → 子网络分化越好）
    avg_diff = np.mean(diffs)
    print(f"\n  平均差异: {avg_diff:.4f}")
    if avg_diff > 0.01:
        print("  → 子网络已经分化，输出视角不同 ✅")
    else:
        print("  → 子网络输出太相似 ❌ 需要不同的初始化或训练策略")
    
    # 基本单元分析
    units_np = units[0].cpu().numpy()  # [4, 28, 28]
    p_map = units_np[3]  # 存在概率
    z_map = units_np[2]  # 深度
    
    print(f"\n{'='*60}")
    print("基本单元分析")
    print(f"{'='*60}")
    print(f"  单元网格: 28×28 = {28*28} 个候选单元")
    print(f"  高存在概率(p>0.5): {(p_map>0.5).sum()} 个")
    print(f"  最高 p: {p_map.max():.4f}, 最低 p: {p_map.min():.4f}")
    print(f"  平均 p: {p_map.mean():.4f}")
    
    # 深度范围
    z_high = z_map[p_map > 0.5]
    if len(z_high) > 0:
        print(f"  高 p 区域深度范围: {z_high.min():.3f} ~ {z_high.max():.3f}")
    
    # 可视化保存
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 保存每个子网络的特征图（平均通道）
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    # 原图
    img_show = cv2.resize(img_rgb, (224, 224))
    axes[0,0].imshow(img_show)
    axes[0,0].set_title("Original")
    
    # 5 个子网络特征（取通道均值）
    for i, (name, feat) in enumerate(zip(names, features)):
        feat_map = feat[0].mean(dim=0).cpu().numpy()
        ax = axes[(i+1)//3, (i+1)%3] if i < 4 else axes[0,1]
        ax.imshow(feat_map, cmap='viridis')
        ax.set_title(f"{name} SubNet")
    
    # 存在概率图
    axes[1,2].imshow(p_map, cmap='hot', vmin=0, vmax=1)
    axes[1,2].set_title("Presence Prob (p)")
    
    # 深度图（只在 p 高的区域显示）
    z_display = z_map.copy()
    z_display[p_map < 0.3] = 0
    axes[1,3].imshow(z_display, cmap='coolwarm')
    axes[1,3].set_title("Depth (z) where p>0.3")
    
    for ax in axes.ravel():
        ax.axis('off')
    
    out_path = str(output_dir / f"five_subnets_{Path(args.image).stem}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  结果图: {out_path}")
    
    print(f"\n{'='*60}")
    print("目前是随机权重，子网络差异仅来自初始化种子")
    print("需要训练后才会真正学到有意义的维度")
    print(f"{'='*60}")


if __name__ == "__main__":
    test()
