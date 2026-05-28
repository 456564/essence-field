"""
单张测试：实时中位数分离物体/背景

每张图独立算范数中位数 → 物体/背景分离
不依赖任何固定阈值。
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

# ─── 测试多张不同图片 ───
img_paths = [
    ("杯子", "test_maccup.png"),
]

DATA = Path("data/caltech101/101_ObjectCategories")
for name, cat in [("椅子", "chair"), ("相机", "camera"), ("蝴蝶", "butterfly"),
                   ("飞机", "airplanes"), ("手表", "watch")]:
    files = sorted((DATA / cat).glob("*.*"))
    if files:
        img_paths.append((name, str(files[0])))

fig, axes = plt.subplots(len(img_paths), 4, figsize=(16, 4 * len(img_paths)))

for row, (img_name, img_path) in enumerate(img_paths):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))

    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        field = pipe(x)

    norms = field[0].norm(dim=0)  # [H, W]
    median = norms.median()
    obj_mask = norms > median

    bg_mean = norms[norms <= median].mean().item()
    fg_mean = norms[norms > median].mean().item()
    sep_ratio = (fg_mean - bg_mean) / bg_mean

    print(f"{img_name}: 背景范数={bg_mean:.2f}  物体范数={fg_mean:.2f}  "
          f"分离度={sep_ratio:.1f}")

    # 可视化
    ax = axes[row, 0]
    ax.imshow(img_small)
    ax.set_title(f"{img_name}（原图）", fontsize=9)
    ax.axis('off')

    ax = axes[row, 1]
    norm_display = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    ax.imshow(norm_display.cpu().numpy(), cmap='hot')
    ax.set_title("活跃度热力图（实时范数）", fontsize=9)
    ax.axis('off')

    ax = axes[row, 2]
    ax.imshow(obj_mask.cpu().numpy().astype(float), cmap='gray')
    ax.set_title(f"物体掩码（中位数阈值）", fontsize=9)
    ax.axis('off')

    ax = axes[row, 3]
    alpha = 0.5
    overlay = (1-alpha) * img_small / 255.0
    mask_color = np.zeros((SIZE, SIZE, 3))
    mask_color[obj_mask.cpu().numpy()] = [0, 0.8, 0.2]
    overlay += alpha * mask_color
    overlay = np.clip(overlay, 0, 1)
    ax.imshow(overlay)
    ax.set_title(f"叠加（分离度={sep_ratio:.1f}）", fontsize=9)
    ax.axis('off')

out = "test_output/single_test_trained.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n结果图: {out}")
