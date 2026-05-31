"""Particle simulation test — cup vs synthetic"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import PhysicalOperatorLayer
from src.essence_space import EssenceSpace
from src.simulation import ParticleSimulator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def run_sim(img_hwc, name, n_particles=300):
    """Run full pipeline and simulation"""
    H, W = img_hwc.shape[:2]
    x_t = torch.from_numpy(img_hwc.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    ops = PhysicalOperatorLayer().to(DEVICE)
    space = EssenceSpace.from_image(x_t, ops)

    sim = ParticleSimulator(num_particles=n_particles, max_steps=max(300, H))
    fx, fy, active, frames = sim.simulate(space)
    retention, trapped = sim.compute_retention(fx, fy, active, space)  # uses cavity_mask now

    print(f"  {name}: retention={retention:.4f}  ({trapped.sum()}/{n_particles} trapped)")

    # Visualization
    img_u8 = (np.clip(img_hwc,0,1)*255).astype(np.uint8)
    wall = space.wall_mask[0,0].cpu().numpy()
    fx_np = fx.cpu().numpy()
    fy_np = fy.cpu().numpy()
    active_np = active.cpu().numpy()
    trapped_np = trapped.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original + wall overlay
    axes[0].imshow(img_u8)
    axes[0].imshow(wall, cmap='Reds', alpha=0.3)
    axes[0].set_title(f'{name}\nWall mask (red)'); axes[0].axis('off')

    # 2. Final particle positions
    axes[1].imshow(img_u8)
    active_mask = active_np
    trapped_mask = trapped_np.cpu().numpy() if isinstance(trapped_np, torch.Tensor) else trapped_np
    out_of_bounds = (fy_np >= H) | (fy_np < 0) | (fx_np < 0) | (fx_np >= W)
    for i in range(len(fx_np)):
        if out_of_bounds[i]:
            continue
        if trapped_mask[i]:
            axes[1].scatter(fx_np[i], fy_np[i], c='lime', s=2, alpha=0.7)
        elif active_mask[i]:
            axes[1].scatter(fx_np[i], fy_np[i], c='cyan', s=2, alpha=0.5)
    axes[1].set_title(f'Particles\nGreen=trapped Cyan=active Red=stuck')
    axes[1].axis('off')

    # 3. Particle density heatmap
    heatmap = np.zeros((H, W), dtype=np.float32)
    for i in range(len(fx_np)):
        xi = int(np.clip(fx_np[i], 0, W-1))
        yi = int(np.clip(fy_np[i], 0, H-1))
        heatmap[yi, xi] += 1
    heatmap = heatmap / max(heatmap.max(), 1)
    axes[2].imshow(img_u8, alpha=0.5)
    axes[2].imshow(heatmap, cmap='hot', alpha=0.7)
    axes[2].set_title(f'Density heatmap\nretention={retention:.3f}')
    axes[2].axis('off')

    # 4. Ju field
    ju = space.get('ju')[0,0].cpu().numpy()
    axes[3].imshow(ju, cmap='hot', vmin=0, vmax=1)
    axes[3].set_title(f'Ju (enclosure)\nmean={ju.mean():.3f}')
    axes[3].axis('off')

    plt.suptitle(f'Particle Simulation: {name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = OUT / f'sim_{name}.png'
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return retention

# ── Test images ──
root = Path(__file__).resolve().parent.parent.parent
tests = []

# 1. Closed rectangle
S = 128
img = np.ones((S,S,3), dtype=np.float32) * 0.5
img[25:103, 25:103] = [0.9, 0.9, 0.9]
tests.append(('closed_rect', img))

# 2. Real cup
cup = cv2.imread(str(root/'test_maccup.png'))
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
h,w = cup.shape[:2]; s = 256/max(h,w)
cup = cv2.resize(cup, (int(w*s), int(h*s)))
tests.append(('cup', cup))

# 3. Caltech images
caltech = root / 'data' / 'caltech101' / '101_ObjectCategories'
if caltech.exists():
    import random
    for cat in random.sample([d for d in caltech.iterdir()
                               if d.is_dir() and d.name!='BACKGROUND_Google'], 4):
        imgs = list(cat.glob('*.jpg'))
        if imgs:
            p = random.choice(imgs)
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
            h,w = img.shape[:2]; s = 256/max(h,w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
            tests.append((p.stem, img))

print(f"{'Name':<18} {'Retention':>12}")
print("-"*32)
for name, img in tests:
    r = run_sim(img, name, n_particles=300)
    print(f"{name:<18} {r:12.4f}")
print(f"\nDone. Images in {OUT}/")
