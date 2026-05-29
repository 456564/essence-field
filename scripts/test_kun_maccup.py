"""测试新坤：纯色杯子→蓝色背景，看杯身内部是否完整"""
import sys, torch
sys.path.insert(0, '.')
from src.pipeline import BaguaPipeline
import cv2, numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BaguaPipeline().to(device).eval()
ckpt = torch.load('checkpoints_fixedcolor/bootstrap_epoch15.pth', map_location=device)
model.fusion.A.data = ckpt['A']
model.operator_layer.projections.load_state_dict(ckpt['proj'])

SIZE = 128

# 读取 test_maccup.png
img = cv2.imread('test_maccup.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = torch.from_numpy(img_rgb).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

with torch.no_grad():
    field = model(x)
    base = model.operator_layer.base_ops(x)

norms = field[0].norm(dim=0).cpu().numpy()

# 可视化
fig, axes = plt.subplots(2, 5, figsize=(16, 7))

# 原图
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("原图（蓝背景+白杯子）")

# 范数热力图
norms_np = norms
vmin, vmax = np.percentile(norms_np, 5), np.percentile(norms_np, 98)
axes[0, 1].imshow(norms_np, cmap='hot', vmin=vmin, vmax=vmax)
axes[0, 1].set_title(f"64维场范数\n中位数={norms_np[50:78,50:78].mean():.2f}")

# 掩码
median = np.median(norms_np)
mask = (norms_np > median).astype(float)
axes[0, 2].imshow(mask, cmap='gray')
sep = (norms_np[mask>0].mean() - norms_np[mask==0].mean()) / (norms_np[mask==0].mean() + 1e-6)
axes[0, 2].set_title(f"物体掩码\n分离度={sep:.1f}, 内部占比={mask[50:78,50:78].mean()*100:.0f}%")

# 各算子输出
op_names = ['qian', 'kun', 'zhen', 'xun', 'kan', 'li', 'gen', 'dui']
for i, name in enumerate(op_names):
    row, col = divmod(i, 4)
    v = base[name][0, 0].cpu().numpy()
    v_disp = v
    ax = axes[row+1, col]
    ax.imshow(v_disp, cmap='hot', vmin=np.percentile(v, 5), vmax=np.percentile(v, 98))
    ax.set_title(name, fontsize=9)
    ax.axis('off')

for i in range(8, 8):
    pass  # already covered

# 隐藏多余轴
axes[1, 4].axis('off')

out = "test_output/test_kun_container.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
plt.close()
print(f"结果: {out}")

# 关键指标
print(f"\n杯子内部（中心 28×28 区域）:")
print(f"  范数均值:   {norms_np[50:78, 50:78].mean():.2f}")
print(f"  掩码占比:   {mask[50:78, 50:78].mean()*100:.0f}%")
print(f"  Kun均值:    {base['kun'][0,0,50:78,50:78].mean():.2f}")
print(f"  分离度:     {sep:.1f}")
