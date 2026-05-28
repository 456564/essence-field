"""
通用性测试（训练后）

训练后的 A 核 + 投影层 → 范数分离物体/背景
看训练后通过率是否提升。
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

# 加载训练后权重
ckpt = torch.load("checkpoints/bootstrap_epoch20.pth", map_location=device)
pipe.fusion.A.data = ckpt['A']
pipe.operator_layer.projections.load_state_dict(ckpt['proj'])
print("加载了训练后的权重")

DATA = Path("data/caltech101/101_ObjectCategories")
all_cats = sorted([c.name for c in DATA.iterdir() if c.is_dir()])
all_cats = [c for c in all_cats if c != "BACKGROUND_Google"]

SIZE = 112

results = []
# 每类只测第一张，共 100 类
n_total = 0
n_pass = 0

# 可视化：每类一行（原图 + 热力图 + 物体掩码）
n_display = 50
fig, axes = plt.subplots(n_display, 3, figsize=(14, n_display * 2))

for row, cat in enumerate(all_cats[:n_display]):
    files = sorted((DATA / cat).glob("*.*"))
    if not files:
        continue
    
    img = cv2.imread(str(files[0]))
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        field = pipe(x)
    
    norms = field[0].norm(dim=0)  # [H, W]
    median = norms.median()
    obj_mask = norms > median
    obj_pct = obj_mask.sum().item() / (SIZE*SIZE) * 100
    
    # 背景/物体范数统计
    bg_norm = norms[norms <= median].mean().item()
    fg_norm = norms[norms > median].mean().item()
    sep_ratio = (fg_norm - bg_norm) / bg_norm
    
    n_total += 1
    if sep_ratio > 2.0:
        n_pass += 1
    
    results.append({
        'cat': cat,
        'obj_pct': obj_pct,
        'bg_norm': bg_norm,
        'fg_norm': fg_norm,
        'sep_ratio': sep_ratio,
    })
    
    cat_label = cat.replace("_", " ")[:14]
    
    # 原图
    ax = axes[row, 0]
    ax.imshow(img_small)
    ax.set_title(cat_label, fontsize=6)
    ax.axis('off')
    
    # 热力图
    ax = axes[row, 1]
    norm_display = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    ax.imshow(norm_display.cpu().numpy(), cmap='hot')
    ax.set_title(f"活跃度", fontsize=6)
    ax.axis('off')
    
    # 掩码
    ax = axes[row, 2]
    mask_vis = obj_mask.float().cpu().numpy()
    ax.imshow(mask_vis, cmap='gray')
    color = 'green' if sep_ratio > 2.0 else 'red'
    ax.set_title(f"物{obj_pct:.0f}% 分{sep_ratio:.1f}",
                 fontsize=6, color=color)
    ax.axis('off')

# 隐藏剩余行
for row in range(len(results), n_display):
    for col in range(3):
        axes[row, col].axis('off')

out = "test_output/generalization_trained2.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"结果图: {out}")

# ─── 统计 ───
print(f"\n{'='*55}")
print(f"通用性测试（训练后）：{n_total} 张")
print(f"{'='*55}")

obj_pcts = [r['obj_pct'] for r in results]
pass_rate = n_pass / n_total * 100

print(f"  分离度>2.0的图片: {n_pass}/{n_total} = {pass_rate:.0f}%")
print(f"  物体占比: mean={np.mean(obj_pcts):.1f}%  "
      f"min={min(obj_pcts):.1f}%  max={max(obj_pcts):.1f}%")

# 分离度统计
sep_ratios = [r['sep_ratio'] for r in results]
print(f"\n  分离度(物体/背景-1):")
print(f"    mean={np.mean(sep_ratios):.1f}  "
      f"min={min(sep_ratios):.1f}  max={max(sep_ratios):.1f}")

# 低分离度的案例
lows = [r for r in results if r['sep_ratio'] < 2.0]
if lows:
    print(f"\n  低分离度 (<2.0) 共 {len(lows)} 个:")
    for r in lows[:15]:
        print(f"    {r['cat']}: 分离度{r['sep_ratio']:.1f}  "
              f"背景{r['bg_norm']:.1f} 物体{r['fg_norm']:.0f}")
