"""
直接读场：64 维向量范数 = 网络活跃度 = 物体位置

不对向量做任何运算，直接看每个像素的 64 维向量多"强"。
强 = 网络高度激活 = 有结构 = 物体
弱 = 网络低激活 = 平坦 = 背景/空
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

# 加载训练后的权重
ckpt = torch.load("checkpoints/bootstrap_epoch20.pth", map_location=device)
pipe.fusion.W_up.data = ckpt.get("W_up", ckpt.get("A")); pipe.fusion.W_dn.data = ckpt.get("W_dn", ckpt.get("A"))
pipe.operator_layer.projections.load_state_dict(ckpt['proj'])
print("加载了训练后的权重")

# ─── 杯子 ───
img = cv2.imread("test_maccup.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_small = cv2.resize(img_rgb, (224, 224))

x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
with torch.no_grad():
    hex_field = pipe(x)  # [1, 64, H, W]

field = hex_field[0]  # [64, H, W]

# ─── 每个像素的 64 维向量范数 = 活跃度 ───
activity = field.norm(dim=0)  # [H, W]

# ─── 活跃度最高的像素取出来 ───
H, W = field.shape[1:]
flat_activity = activity.view(-1)
# 前 20% 活跃的像素
top_k = int(flat_activity.numel() * 0.2)
top_idx = torch.topk(flat_activity, top_k).indices
top_mask = torch.zeros(H * W, dtype=torch.bool)
top_mask[top_idx] = True
top_mask = top_mask.view(H, W)

# ─── 可视化 ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. 原图
ax = axes[0, 0]
ax.imshow(img_small)
ax.set_title("原图")
ax.axis('off')

# 2. 活跃度热力图
ax = axes[0, 1]
act_np = activity.cpu().numpy()
ax.imshow(act_np, cmap='hot')
ax.set_title("64维向量范数（越亮=网络越活跃）")
ax.axis('off')

# 3. 活跃度分布直方图
ax = axes[0, 2]
ax.hist(flat_activity.cpu().numpy(), bins=100)
ax.axvline(x=flat_activity.median().item(), color='r', linestyle='--',
           label=f"中位数={flat_activity.median().item():.3f}")
ax.axvline(x=flat_activity.topk(top_k).values.min().item(), color='g', linestyle='--',
           label=f"前20%阈值")
ax.set_xlabel("活跃度（向量范数）")
ax.set_ylabel("像素数")
ax.legend(fontsize=8)
ax.set_title("活跃度分布")

# 4. 前 20% 活跃的像素（二值图）
ax = axes[1, 0]
ax.imshow(top_mask.cpu().numpy().astype(float), cmap='gray')
ax.set_title("前20%活跃像素（白=高活跃）")
ax.axis('off')

# 5. 活跃度叠加原图
ax = axes[1, 1]
alpha = 0.6
overlay = (1-alpha) * img_small / 255.0 + alpha * plt.cm.hot(act_np / act_np.max())[:, :, :3]
overlay = np.clip(overlay, 0, 1)
ax.imshow(overlay)
ax.set_title("活跃度叠加原图")
ax.axis('off')

# 6. cross-section：取第 100 行看活跃度
ax = axes[1, 2]
row = H // 2
row_act = activity[row, :].cpu().numpy()
ax.plot(range(W), row_act)
ax.set_title(f"第 {row} 行活跃度截面")
ax.set_xlabel("像素")
ax.set_ylabel("活跃度")

out = "test_output/field_activity.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n结果图: {out}")
print()
print("观察：")
print("  第2张图：最亮的区域是否对应物体本身？")
print("  第4张图：前20%活跃像素构成什么？物体轮廓？还是整个物体？")
print("  背景区域是否显著暗于物体区域？")
