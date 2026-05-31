"""通用性测试：8算子 + 66维场范数，多类别多图片"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np, matplotlib.pyplot as plt, random
from pathlib import Path
from src.operators import PhysicalOperatorLayer
from src.pipeline import PhysicalPipeline

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
op_layer = PhysicalOperatorLayer()
pipe = PhysicalPipeline().to(device).eval()

# 从 Caltech101 随机选不同类别的图片
DATA = "data/caltech101/101_ObjectCategories"
categories = ['cup', 'chair', 'watch', 'lamp',
              'Motorbikes', 'airplanes', 'Faces', 'Leopards', 'starfish']

random.seed(42)
test_images = []
for cat in categories:
    cat_dir = Path(DATA) / cat
    if cat_dir.exists():
        imgs = list(cat_dir.glob("*.jpg"))
        if imgs:
            test_images.append(str(random.choice(imgs)))

# 不要再加自定义图片，正好9张

print(f"测试 {len(test_images)} 张图片（{len(categories)} 类）")

results = []
fig, axes = plt.subplots(len(test_images), 6, figsize=(22, 4 * len(test_images)))

for row, img_path in enumerate(test_images):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    x = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    img_norm = img_resized / 255.0

    with torch.no_grad():
        base = op_layer(x)
        field = pipe(x)

    norms = field[0].norm(dim=0).cpu().numpy()
    median = np.median(norms)
    sep = (norms[norms>median].mean() - norms[norms<=median].mean()) / (norms[norms<=median].mean() + 1e-6)
    mask = (norms > median).astype(float)

    ju_map = base[0, 4].cpu().numpy()
    vp_map = base[0, 8].cpu().numpy()
    yang_map = base[0, 6].cpu().numpy()

    results.append({'name': Path(img_path).stem, 'sep': sep, 'mask_ratio': mask.mean()})

    # (row, 0) 原图
    axes[row, 0].imshow(img_resized)
    axes[row, 0].set_title(Path(img_path).stem, fontsize=8)
    axes[row, 0].axis('off')

    # (row, 1) ju 亮绿
    ju_color = np.zeros((224, 224, 3))
    ju_color[..., 1] = np.clip((ju_map - ju_map.min()) / (ju_map.max() - ju_map.min() + 1e-8), 0, 1)
    axes[row, 1].imshow(img_norm * 0.3 + ju_color * 0.7)
    axes[row, 1].set_title(f"ju(围合) {ju_map.mean():.3f}", fontsize=8, color='green')
    axes[row, 1].axis('off')

    # (row, 2) void 亮红
    vp_color = np.zeros((224, 224, 3))
    vp_color[..., 0] = np.clip((vp_map - vp_map.min()) / (vp_map.max() - vp_map.min() + 1e-8), 0, 1)
    axes[row, 2].imshow(img_norm * 0.3 + vp_color * 0.7)
    axes[row, 2].set_title(f"void(虚空) {vp_map.mean():.3f}", fontsize=8, color='red')
    axes[row, 2].axis('off')

    # (row, 3) yang 亮蓝
    yang_color = np.zeros((224, 224, 3))
    yang_color[..., 2] = np.clip((yang_map - yang_map.min()) / (yang_map.max() - yang_map.min() + 1e-8), 0, 1)
    axes[row, 3].imshow(img_norm * 0.3 + yang_color * 0.7)
    axes[row, 3].set_title(f"yang(实体) {yang_map.mean():.3f}", fontsize=8, color='blue')
    axes[row, 3].axis('off')

    # (row, 4) ju+void 双色
    dual = np.zeros((224, 224, 3))
    ju_norm = np.clip((ju_map - ju_map.min()) / (ju_map.max() - ju_map.min() + 1e-8), 0, 1)
    dual[..., 1] = ju_norm * 0.8
    dual[..., 0] = np.clip((vp_map - vp_map.min()) / (vp_map.max() - vp_map.min() + 1e-8), 0, 1) * 0.8
    axes[row, 4].imshow(img_norm * 0.25 + dual * 0.75)
    axes[row, 4].set_title("ju(绿)+void(红)", fontsize=8)
    axes[row, 4].axis('off')

    # (row, 5) 掩码 + 范数叠加
    vmin, vmax = np.percentile(norms, 5), np.percentile(norms, 95)
    norms_norm = np.clip((norms - vmin) / (vmax - vmin + 1e-8), 0, 1)
    norms_color = plt.cm.jet(norms_norm)[:, :, :3]
    blended_n = img_norm * 0.3 + norms_color * 0.7
    axes[row, 5].imshow(blended_n)
    axes[row, 5].set_title(f"范数叠加 sep={sep:.1f}", fontsize=8)
    axes[row, 5].axis('off')

plt.tight_layout()
out = "test_output/generalization_vivid.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()

# 统计
seps = [r['sep'] for r in results]
print(f"\n{'='*50}")
print(f"通用性测试统计")
print(f"{'='*50}")
print(f"{'图片':<20} {'分离度':>8} {'物体占比':>8}")
print("-"*40)
for r in results:
    print(f"{r['name']:<20} {r['sep']:>8.1f} {r['mask_ratio']*100:>7.0f}%")
print("-"*40)
print(f"平均分离度: {np.mean(seps):.1f}")
print(f">2.0: {sum(s>2 for s in seps)}/{len(seps)}")
print(f"结果图: {out}")
