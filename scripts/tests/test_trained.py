"""
验证训练效果：加载训练后的 A + 投影层，对比训练前后的范数分离
"""

import sys, torch, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import BaguaPipeline
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 加载训练后的模型 ───
model_trained = BaguaPipeline(d=8).to(device)
checkpoint = torch.load("checkpoints/bootstrap_epoch20.pth", map_location=device)
model_trained.fusion.W_up.data = checkpoint.get("W_up", checkpoint.get("A")); model_trained.fusion.W_dn.data = checkpoint.get("W_dn", checkpoint.get("A"))
model_trained.operator_layer.projections.load_state_dict(checkpoint['proj'])
print("加载了训练后的权重")

# ─── 随机权重的模型（对比） ───
model_random = BaguaPipeline(d=8).to(device)
print("随机权重（未训练）")

# ─── 测试图片 ───
img_paths = [
    ("杯子", "test_maccup.png"),
]

DATA = Path("data/caltech101/101_ObjectCategories")
for name, cat in [("椅子", "chair"), ("相机", "camera"), ("蝴蝶", "butterfly"),
                   ("飞机", "airplanes"), ("手表", "watch"), ("摩托车", "Motorbikes")]:
    files = sorted((DATA / cat).glob("*.*"))
    if files:
        img_paths.append((name, str(files[0])))

SIZE = 128

for model, label in [(model_random, "训练前(随机)"), (model_trained, "训练后")]:
    model.eval()
    print(f"\n{label}:")
    
    bg_vals = []
    fg_vals = []
    sep_ratios = []
    dir_consistency = []  # 物体内部方向一致性
    
    for img_name, img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, (SIZE, SIZE))
        
        x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            field = model(x)
        
        field_flat = field[0].view(64, -1).t()
        norms = field_flat.norm(dim=1).cpu().numpy()
        median = np.median(norms)
        
        # 范数分离
        bg_mean = norms[norms <= median].mean()
        fg_mean = norms[norms > median].mean()
        sep_ratio = (fg_mean - bg_mean) / bg_mean
        bg_vals.append(bg_mean)
        fg_vals.append(fg_mean)
        sep_ratios.append(sep_ratio)
        
        # 方向一致性：物体内部所有像素两两之间的平均余弦相似度
        fg_vecs = field_flat[norms > median]
        fg_normed = fg_vecs / (fg_vecs.norm(dim=1, keepdim=True) + 1e-8)
        n_sample = min(200, fg_normed.shape[0])
        idx = torch.randperm(fg_normed.shape[0])[:n_sample]
        sampled = fg_normed[idx]
        sim = (sampled @ sampled.t()).mean().item()
        dir_consistency.append(sim)
    
    print(f"  背景范数均值: {np.mean(bg_vals):.3f}")
    print(f"  物体范数均值: {np.mean(fg_vals):.3f}")
    print(f"  范数分离度(物体/背景-1): {np.mean(sep_ratios):.3f}")
    print(f"  物体内方向一致性(cos): {np.mean(dir_consistency):.3f}")

print(f"\n{'='*55}")
print(f"对比重点：")
print(f"  范数分离度 → 训练前后是否变化？（训练不直接影响范数）")
print(f"  方向一致性 → 训练后是否提升？（训练目标就是这个）")
print(f"{'='*55}")
