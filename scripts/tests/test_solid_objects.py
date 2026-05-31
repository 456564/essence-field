"""Test void_prob on smooth solid objects — hardest case for 2D cavity detection"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import PhysicalOperatorLayer, cu, void_prob
from src.essence_space import EssenceSpace
from src.simulation import ParticleSimulator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def run_one(img_hwc, name, n=200):
    H,W = img_hwc.shape[:2]
    t = torch.from_numpy(img_hwc.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    ops = PhysicalOperatorLayer().to(DEVICE)
    with torch.no_grad():
        field = ops(t)
    space = EssenceSpace(field)
    vp = space.get('void_prob')[0,0].cpu().numpy()
    c  = cu(t)[0,0].cpu().numpy()
    ju_= space.get('ju')[0,0].cpu().numpy()
    yg = space.get('yang')[0,0].cpu().numpy()

    # Run simulation
    sim = ParticleSimulator(num_particles=n, max_steps=max(300, H))
    fx, fy, active, frames = sim.simulate(space)
    ret, trapped = sim.compute_retention(fx, fy, active, space)

    # Key metrics
    vp_hi = (vp>0.3).mean(); cavity_px = (space.cavity_mask[0,0]>0.5).sum()
    print(f"{name:<20} cu_mean={c.mean():.3f} vp>0.3={vp_hi:.2f} cavity_px={cavity_px} ju_mean={ju_.mean():.3f} yang_mean={yg.mean():.3f} retention={ret:.4f}")

    # Visualize
    img_u8 = (np.clip(img_hwc,0,1)*255).astype(np.uint8)
    wall = space.wall_mask[0,0].cpu().numpy()
    fig, axes = plt.subplots(1, 6, figsize=(26, 4))
    axes[0].imshow(img_u8); axes[0].set_title(name, fontsize=8); axes[0].axis('off')
    axes[1].imshow(ju_, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Ju\nmean={ju_.mean():.3f}', fontsize=8); axes[1].axis('off')
    axes[2].imshow(c, cmap='hot', vmin=0, vmax=0.1)
    axes[2].set_title(f'Cu\nmean={c.mean():.3f}', fontsize=8); axes[2].axis('off')
    axes[3].imshow(yg, cmap='hot', vmin=0, vmax=1)
    axes[3].set_title(f'Yang\nmean={yg.mean():.3f}', fontsize=8); axes[3].axis('off')
    axes[4].imshow(vp, cmap='hot', vmin=0, vmax=1)
    axes[4].set_title(f'VoidProb\nmean={vp.mean():.3f}', fontsize=8); axes[4].axis('off')
    # Particles
    axes[5].imshow(img_u8)
    fx_np=fx.cpu().numpy(); fy_np=fy.cpu().numpy()
    tr=trapped.cpu().numpy() if isinstance(trapped, torch.Tensor) else trapped
    ac=active.cpu().numpy()
    for i in range(len(fx_np)):
        if fy_np[i]>=H or fx_np[i]>=W or fy_np[i]<0 or fx_np[i]<0: continue
        if tr[i]: axes[5].scatter(fx_np[i],fy_np[i],c='lime',s=2,alpha=0.6)
        elif ac[i]: axes[5].scatter(fx_np[i],fy_np[i],c='cyan',s=2,alpha=0.4)
    axes[5].set_title(f'Particles ret={ret:.2f}', fontsize=8); axes[5].axis('off')

    plt.suptitle(f'{name}: cu={c.mean():.3f} void>0.3={vp_hi:.2f} retention={ret:.2f}', fontsize=10)
    plt.tight_layout()
    out=OUT/f'solid_{name}.png'
    fig.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    return ret, vp_hi

root = Path(__file__).resolve().parent.parent.parent
caltech = root / 'data' / 'caltech101' / '101_ObjectCategories'
files = []

# Target categories for smooth solid objects
targets = ['apple', 'egg', 'ball', 'tomato', 'cannon', 'buddha', 'cup', 'binocular',
           'camera', 'brain', 'cellphone', 'chair', 'dolphin', 'elephant',
           'garfield', 'hedgehog', 'kangaroo', 'lamp', 'lotus',
           'mayfly', 'nautilus', 'octopus', 'pizza', 'platypus',
           'rhino', 'scorpion', 'snoopy', 'stapler', 'tick', 'umbrella',
           'watch', 'wrench', 'yin_yang']

if caltech.exists():
    for cat in [d for d in caltech.iterdir() if d.is_dir() and d.name in targets]:
        imgs = list(cat.glob('*.jpg'))
        if imgs:
            p = imgs[0]  # first image
            img = cv2.imread(str(p))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
            h,w = img.shape[:2]; s = 256/max(h,w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
            files.append((cat.name, img))

# Add synthetic
S=128
img_rect = np.ones((S,S,3), dtype=np.float32)*0.5; img_rect[25:103,25:103]=[0.9,0.9,0.9]
files.insert(0, ('closed_rect', img_rect))
img_cup = cv2.imread(str(root/'test_maccup.png'))
img_cup = cv2.cvtColor(img_cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
h,w = img_cup.shape[:2]; s=256/max(h,w)
files.insert(1, ('cup', cv2.resize(img_cup, (int(w*s),int(h*s)))))

print(f"{'Name':<18} {'cuMean':>8} {'vp>0.3':>8} {'cavPx':>8} {'juMean':>8} {'ygMean':>8} {'ret':>8}")
print("-"*80)
results = []
for name, img in files:
    ret, vphi = run_one(img, name, n=200)
    results.append((name, ret, vphi))

# Summary
print(f"\n--- FAILURES (retention > 0.3 on non-containers) ---")
for name, ret, vphi in results:
    is_container = name in ['cup', 'closed_rect', 'lotus', 'pizza']
    if ret > 0.3 and not is_container:
        print(f"  {name}: ret={ret:.3f} vp>0.3={vphi:.2f}  ← FALSE CONTAINER")
print(f"\nDone. Images in {OUT}/solid_*.png")
