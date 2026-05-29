"""
RGB 活跃度热力图

不是把 64 维向量坍缩成一个标量范数。
而是分别看 R、G、B 三通道各自对活跃度的贡献。
"""

import sys, torch, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import BaguaPipeline
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = BaguaPipeline(d=8).to(device).eval()

ckpt = torch.load("checkpoints_colormod/bootstrap_epoch20.pth", map_location=device)
pipe.fusion.A.data = ckpt['A']
pipe.operator_layer.projections.load_state_dict(ckpt['proj'])
print("加载了训练后的权重")

SIZE = 224

# 测试图
DATA = Path("data/caltech101/101_ObjectCategories")
for cat in ["butterfly", "leopards", "crab", "cup"]:
    files = sorted((DATA / cat).glob("*.*"))
    if files:
        break

img = cv2.imread(str(files[0]))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_small = cv2.resize(img_rgb, (SIZE, SIZE))

# ─── 分别对 R、G、B 通道跑 64 维场 ───
channel_norms = []
for c in range(3):
    ch_img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    ch_img[:, :, c] = img_small[:, :, c] / 255.0
    x = torch.from_numpy(ch_img).permute(2,0,1).float().unsqueeze(0).to(device)
    with torch.no_grad():
        field = pipe(x)
    ch_norm = field[0].norm(dim=0)  # [H, W]
    channel_norms.append(ch_norm)

# 合并为 [3, H, W] — RGB 三通道各自的活跃度
act_rgb = torch.stack(channel_norms, dim=0)  # [3, H, W]
total_act = act_rgb.sum(dim=0)  # [H, W]

# 归一化到 0~1
act_normed = act_rgb / (act_rgb.max() + 1e-8)

# 用 RGB 三通道活跃度直接着色的热力图
rgb_heat = act_normed.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

# ─── 可视化 ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 原图
ax = axes[0, 0]
ax.imshow(img_small)
ax.set_title("原图")
ax.axis('off')

# 传统灰度热力图
ax = axes[0, 1]
total_np = total_act.cpu().numpy()
ax.imshow(total_np, cmap='hot')
ax.set_title("传统热力图（标量范数）")
ax.axis('off')

# RGB 活跃度热力图
ax = axes[0, 2]
ax.imshow(rgb_heat)
ax.set_title("RGB 活跃度热力图\n（R/G/B 各通道活跃度着RGB色）")
ax.axis('off')

# 优势通道图
ax = axes[1, 0]
dominant = act_rgb.argmax(dim=0).cpu().numpy()  # [H, W]
dom_colors = np.zeros((SIZE, SIZE, 3))
dom_colors[dominant == 0] = [1, 0, 0]
dom_colors[dominant == 1] = [0, 1, 0]
dom_colors[dominant == 2] = [0, 0, 1]
# 三个通道都弱的不着色
low_mask = total_np < total_np.max() * 0.1
dom_colors[low_mask] = [0, 0, 0]
ax.imshow(dom_colors)
ax.set_title("优势通道\n（红=R, 绿=G, 蓝=B, 黑=背景）")
ax.axis('off')

# 叠加原图
ax = axes[1, 1]
alpha = 0.5
overlay = (1-alpha) * img_small / 255.0 + alpha * rgb_heat
overlay = np.clip(overlay, 0, 1)
ax.imshow(overlay)
ax.set_title("RGB 活跃度叠加原图")
ax.axis('off')

# 逐通道活跃度 vs 像素位置（取中间一行）
ax = axes[1, 2]
row = SIZE // 2
for c, name, color in [(0, 'R', 'red'), (1, 'G', 'green'), (2, 'B', 'blue')]:
    ax.plot(range(SIZE), act_rgb[c, row, :].cpu().numpy(), 
            color=color, label=name, alpha=0.7)
ax.legend(fontsize=8)
ax.set_title(f"第{row}行：逐通道活跃度")
ax.set_xlabel("像素")
ax.set_ylabel("活跃度")
ax.grid(True, alpha=0.3)

out = "test_output/rgb_heatmap.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n结果图: {out}")
print()
print("观察：")
print("  原始热力图 vs RGB 热力图：RGB 版是否提供了更多信息？")
print("  优势通道图：物体的不同部分是否由不同颜色通道主导？")
