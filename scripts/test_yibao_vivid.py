"""显眼展示：高饱和度彩色叠加，一眼看出效果"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np, matplotlib.pyplot as plt
from src.operators import PhysicalOperatorLayer
from src.pipeline import PhysicalPipeline

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
op_layer = PhysicalOperatorLayer()
pipe = PhysicalPipeline().to(device).eval()

img = cv2.imread('test_yibao.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
x = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

with torch.no_grad():
    base = op_layer(x)
    field = pipe(x)

norms = field[0].norm(dim=0).cpu().numpy()
median = np.median(norms)

# 逐算子提取
dong_map = base[0, 0].cpu().numpy()
gang_map = base[0, 1].cpu().numpy()
cu_map   = base[0, 2].cpu().numpy()
rou_map  = base[0, 3].cpu().numpy()
ju_map   = base[0, 4].cpu().numpy()
dist_map = base[0, 5].cpu().numpy()
yang_map = base[0, 6].cpu().numpy()
yin_map  = base[0, 7].cpu().numpy()
vp_map   = base[0, 8].cpu().numpy()

img_norm = img_resized / 255.0

fig, axes = plt.subplots(3, 4, figsize=(18, 13))

# ── 第1行：关键结果 ──
# (0,0) 原图
axes[0,0].imshow(img_resized); axes[0,0].set_title("原图", fontsize=10, fontweight='bold')
axes[0,0].axis('off')

# (0,1) 围合(ju) 亮绿色叠加
ju_color = np.zeros((224, 224, 3))
ju_color[..., 1] = np.clip((ju_map - ju_map.min()) / (ju_map.max() - ju_map.min() + 1e-8), 0, 1)
blended = img_norm * 0.3 + ju_color * 0.7
axes[0,1].imshow(blended)
axes[0,1].set_title("ju(围合) 亮绿叠加", fontsize=10, fontweight='bold', color='green')
axes[0,1].axis('off')

# (0,2) void_prob 亮红叠加
vp_color = np.zeros((224, 224, 3))
vp_color[..., 0] = vp_map  # R通道
blended_vp = img_norm * 0.3 + vp_color * 0.7
axes[0,2].imshow(blended_vp)
axes[0,2].set_title("void_prob(虚空) 亮红叠加", fontsize=10, fontweight='bold', color='red')
axes[0,2].axis('off')

# (0,3) 实体(yang) 亮蓝叠加
yang_color = np.zeros((224, 224, 3))
yang_color[..., 2] = np.clip((yang_map - yang_map.min()) / (yang_map.max() - yang_map.min() + 1e-8), 0, 1)
blended_y = img_norm * 0.3 + yang_color * 0.7
axes[0,3].imshow(blended_y)
axes[0,3].set_title("yang(实体) 亮蓝叠加", fontsize=10, fontweight='bold', color='blue')
axes[0,3].axis('off')

# ── 第2行：围合组合 ──
# (1,0) ju + void_prob 双色叠加（绿=围合，红=虚空）
dual = np.zeros((224, 224, 3))
dual[..., 1] = np.clip((ju_map - ju_map.min()) / (ju_map.max() - ju_map.min() + 1e-8), 0, 1) * 0.8
dual[..., 0] = vp_map * 0.8
blended_dual = img_norm * 0.25 + dual * 0.75
axes[1,0].imshow(blended_dual)
axes[1,0].set_title("ju(绿)+void(红) 合并", fontsize=10, fontweight='bold')
axes[1,0].axis('off')

# (1,1) dong 边缘
dong_color = np.zeros((224, 224, 3))
dong_color[:] = [0, 0, 0]
dong_norm = np.clip((dong_map - dong_map.min()) / (dong_map.max() - dong_map.min() + 1e-8), 0, 1)
dong_color[..., 0] = dong_norm * 0.9
dong_color[..., 1] = dong_norm * 0.3
blended_dong = img_norm * 0.3 + dong_color * 0.7
axes[1,1].imshow(blended_dong)
axes[1,1].set_title("dong(梯度) 红色边缘", fontsize=10, fontweight='bold', color='darkred')
axes[1,1].axis('off')

# (1,2) 范数热力图（高饱和度）
vmin, vmax = np.percentile(norms, 5), np.percentile(norms, 95)
norms_norm = np.clip((norms - vmin) / (vmax - vmin + 1e-8), 0, 1)
axes[1,2].imshow(norms_norm, cmap='jet', vmin=0, vmax=1)
axes[1,2].set_title("66维场范数 (高饱和jet)", fontsize=10, fontweight='bold')
axes[1,2].axis('off')

# (1,3) 范数叠加
norms_color = plt.cm.jet(norms_norm)[:, :, :3]
blended_n = img_norm * 0.3 + norms_color * 0.7
axes[1,3].imshow(blended_n)
axes[1,3].set_title("范数叠加原图", fontsize=10, fontweight='bold')
axes[1,3].axis('off')

# ── 第3行：掩码结果 ──
# (2,0) 中位数掩码（白色物体 + 黑色背景）
mask = (norms > median).astype(float)
mask_color = np.zeros((224, 224, 3))
mask_color[mask > 0] = [0, 1, 0]  # 绿色
blended_mask = img_norm * 0.4 + mask_color * 0.6
axes[2,0].imshow(blended_mask)
sep = (norms[mask>0].mean() - norms[mask==0].mean()) / (norms[mask==0].mean() + 1e-6)
axes[2,0].set_title(f"物体掩码 (分离度{sep:.1f})", fontsize=10, fontweight='bold')
axes[2,0].axis('off')

# (2,1) 对比：原始ju叠加 vs 范数掩码
axes[2,1].imshow(blended_dual)
axes[2,1].set_title("ju(绿)+void(红) → 容器检测", fontsize=10, fontweight='bold')
axes[2,1].axis('off')

# (2,2) 硬边界(gang)叠加
gang_color = np.zeros((224, 224, 3))
gang_norm = np.clip((gang_map - gang_map.min()) / (gang_map.max() - gang_map.min() + 1e-8), 0, 1)
gang_color[:, :, 0] = gang_norm * 0.9
gang_color[:, :, 2] = gang_norm * 0.9
blended_gang = img_norm * 0.3 + gang_color * 0.7
axes[2,2].imshow(blended_gang)
axes[2,2].set_title("gang(硬边界) 品红", fontsize=10, fontweight='bold')
axes[2,2].axis('off')

# (2,3) 文字总结
axes[2,3].text(0.5, 0.5,
    f"test_yibao.jpg 分析结果\n\n"
    f"分离度: {sep:.1f}\n"
    f"物体占比: {mask.mean()*100:.0f}%\n"
    f"ju均值: {ju_map.mean():.3f}\n"
    f"虚空概率: {vp_map.mean():.3f}\n\n"
    f"ju(绿)=围合区域\n"
    f"void(红)=虚空(空腔)\n"
    f"yang(蓝)=实体表面",
    ha='center', va='center', fontsize=12,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[2,3].axis('off')

plt.tight_layout()
out = "test_output/test_yibao_vivid.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ {out}")
print(f"\n关键指标:")
print(f"  分离度: {sep:.1f}")
print(f"  ju(围合) 均值: {ju_map.mean():.3f}")
print(f"  void_prob(虚空) 均值: {vp_map.mean():.3f}")
print(f"  yang(实体) 均值: {yang_map.mean():.3f}")
print(f"  dong(梯度) 均值: {dong_map.mean():.3f}")
