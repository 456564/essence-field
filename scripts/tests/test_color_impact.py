"""
测试颜色空间的物体分离效果

用 HSV 各通道替代 RGB 作为输入，看物体/背景分离度是否提升。
算子在内部转灰度，但不同颜色空间的结构不同→分离度不同。
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

ckpt = torch.load("checkpoints/bootstrap_epoch20.pth", map_location=device)
pipe.fusion.W_up.data = ckpt.get("W_up", ckpt.get("A")); pipe.fusion.W_dn.data = ckpt.get("W_dn", ckpt.get("A"))
pipe.operator_layer.projections.load_state_dict(ckpt['proj'])
print("加载了训练后的权重")

DATA = Path("data/caltech101/101_ObjectCategories")

# 选几张色彩丰富的图
test_imgs = []
for cat in ["butterfly", "leopards", "crab", "sunflower", "lotus", "cup"]:
    files = sorted((DATA / cat).glob("*.*"))
    if files:
        img = cv2.imread(str(files[0]))
        if img is not None:
            test_imgs.append((cat, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

SIZE = 224

# 测试不同的色彩空间表示
color_spaces = {
    "原始RGB": lambda x: x,
    "灰度(3ch)": lambda x: np.stack([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)]*3, axis=2),
}

# 详细测试结果
print(f"{'='*70}")
print(f"{'图片':<12} {'色彩空间':<12} {'背景范数':<10} {'物体范数':<10} {'分离度':<10}")
print(f"{'='*70}")

all_results = []

for img_name, img_rgb in test_imgs:
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    
    for cs_name, cs_fn in color_spaces.items():
        cs_img = cs_fn(img_small).astype(np.float32) / 255.0
        
        x = torch.from_numpy(cs_img).permute(2,0,1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            field = pipe(x)
        
        norms = field[0].norm(dim=0)
        median = norms.median()
        
        bg_n = norms[norms <= median].mean().item()
        fg_n = norms[norms > median].mean().item()
        sep = (fg_n - bg_n) / bg_n
        
        all_results.append((img_name, cs_name, bg_n, fg_n, sep))
        print(f"{img_name:<12} {cs_name:<12} {bg_n:<10.2f} {fg_n:<10.2f} {sep:<10.1f}")

# 汇总：每种色彩空间对物体的平均分离度
print(f"\n{'='*55}")
print(f"汇总：平均分离度（越高越好）")
print(f"{'='*55}")

from collections import defaultdict
space_seps = defaultdict(list)
for img_name, cs_name, bg_n, fg_n, sep in all_results:
    space_seps[cs_name].append(sep)

for cs_name, seps in sorted(space_seps.items()):
    print(f"  {cs_name:<12} 平均分离度={np.mean(seps):.1f}  "
          f"min={min(seps):.1f}  max={max(seps):.1f}")
