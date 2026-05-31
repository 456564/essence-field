"""测试 test_yibao.jpg：8物理算子 + 66维场范数"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np, matplotlib.pyplot as plt
from src.operators import PhysicalOperatorLayer
from src.pipeline import PhysicalPipeline
from src.visualize import essence_panel

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
sep = (norms[norms>median].mean() - norms[norms<=median].mean()) / (norms[norms<=median].mean() + 1e-6)

print(f"分离度: {sep:.1f}")
print(f"范数范围: {norms.min():.2f} ~ {norms.max():.2f}")

# 显示
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
# 原图
axes[0,0].imshow(img_resized); axes[0,0].set_title("原图"); axes[0,0].axis('off')
# 范数
vmin, vmax = np.percentile(norms, 2), np.percentile(norms, 98)
axes[0,1].imshow(norms, cmap='hot', vmin=vmin, vmax=vmax)
axes[0,1].set_title(f"66维场范数 (分离度{sep:.1f})")
axes[0,1].axis('off')
# 掩码
mask = (norms > median).astype(float)
axes[0,2].imshow(mask, cmap='gray')
axes[0,2].set_title(f"掩码 (中位数阈值)")
axes[0,2].axis('off')
# void_prob
vp = base[0, 8].cpu().numpy()
axes[0,3].imshow(vp, cmap='plasma', vmin=0, vmax=1)
axes[0,3].set_title("void_prob (虚空)")
axes[0,3].axis('off')

# 8算子
op_names = ['dong(梯度)', 'gang(边界)', 'cu(纹理)', 'rou(渐变)',
            'ju(围合)', 'dist(距边)', 'yang(实体)', 'yin(虚空)']
for i, ax in enumerate(axes.flat[4:]):
    if i >= 8: break
    spat = base[0, i].cpu().numpy()
    vmin2, vmax2 = np.percentile(spat, 2), np.percentile(spat, 98)
    ax.imshow(spat, cmap='viridis', vmin=vmin2, vmax=vmax2)
    ax.set_title(f"{op_names[i]}  {spat.mean():.3f}")
    ax.axis('off')

plt.tight_layout()
out = "test_output/test_yibao.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"结果图: {out}")
