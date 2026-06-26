"""
64 维向量的两层信息：
  范数 = 物体在哪（活跃度）
  方向 = 物质是什么

先用范数找到物体，再在物体内部用方向区分部件。
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

img = cv2.imread("test_maccup.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_small = cv2.resize(img_rgb, (224, 224))

x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
with torch.no_grad():
    hex_field = pipe(x)

field = hex_field[0]  # [64, H, W]
H, W = field.shape[1:]

# ─── 第1层：范数 → 找到物体 ───
activity = field.norm(dim=0)  # [H, W]
threshold = activity.median()  # 用中位数做阈值
object_mask = activity >= threshold

# ─── 第2层：方向 → 在物体内部区分物质类型 ───
# 对物体像素做 K-Means（K=3：亮面、阴影、边缘）
object_vecs = field[:, object_mask]  # [64, N]
object_normed = object_vecs / (object_vecs.norm(dim=0, keepdim=True) + 1e-8)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init=5)
obj_labels = kmeans.fit_predict(object_normed.t().cpu().numpy())

# 映射回原图
labels_full = np.full((H, W), -1, dtype=int)
obj_indices = torch.where(object_mask)
for i in range(3):
    mask_2d = torch.zeros(H, W, dtype=torch.bool)
    mask_2d[obj_indices[0][obj_labels == i], obj_indices[1][obj_labels == i]] = True
    labels_full[mask_2d] = i

# ─── 可视化 ───
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. 原图
ax = axes[0, 0]
ax.imshow(img_small)
ax.set_title("原图")
ax.axis('off')

# 2. 物体位置（范数>中位数）
ax = axes[0, 1]
ax.imshow(object_mask.cpu().numpy().astype(float), cmap='gray')
ax.set_title("第1层：物体位置（活跃度>中位数）")
ax.axis('off')

# 3. 物体边界（去掉物体的外边缘，只剩内部）
ax = axes[0, 2]
# 物体区域的梯度 → 物体内部的细分边界
obj_np = object_mask.cpu().numpy()
from scipy.ndimage import binary_erosion
eroded = binary_erosion(obj_np, iterations=2)
interior = obj_np & ~eroded  # 边界
ax.imshow(interior.astype(float), cmap='gray')
ax.set_title("物体外部轮廓")
ax.axis('off')

# 4. 第2层：物体内部的物质类型
ax = axes[1, 0]
seg = np.zeros((H, W, 3))
cmap = plt.cm.Set1
for i in range(3):
    seg[labels_full == i] = cmap(i/3)[:3]
# 背景 = 黑色
seg[labels_full == -1] = [0, 0, 0]
ax.imshow(seg)
ax.set_title("第2层：物体内部物质类型\n（方向聚类 K=3）")
ax.axis('off')

# 5. 叠加
ax = axes[1, 1]
alpha = 0.6
overlay = (1-alpha) * img_small / 255.0 + alpha * seg
overlay = np.clip(overlay, 0, 1)
ax.imshow(overlay)
ax.set_title("叠加原图")
ax.axis('off')

# 6. 每个物质类型的 64 维方向签名
ax = axes[1, 2]
# 取每个聚类中心
centers = kmeans.cluster_centers_  # [3, 64]
ax.imshow(centers, aspect='auto', cmap='coolwarm', vmin=-0.3, vmax=0.6)
ax.set_yticks(range(3))
ax.set_yticklabels([f"类型{i}" for i in range(3)], fontsize=8)
ax.set_xlabel("64维向量方向")
ax.set_title("各物质类型的本质签名")

out = "test_output/two_layer_readout.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n结果图: {out}")
