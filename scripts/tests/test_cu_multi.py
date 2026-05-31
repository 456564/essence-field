"""Cu operator multi-image test"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2, random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import cu

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process(img):
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    return cu(t)[0,0].cpu().numpy()

root = Path(__file__).resolve().parent.parent.parent
caltech = root / 'data' / 'caltech101' / '101_ObjectCategories'
files = []
if caltech.exists():
    for cat in random.sample([d for d in caltech.iterdir() if d.is_dir() and d.name!='BACKGROUND_Google'], 8):
        imgs = list(cat.glob('*.jpg'))
        if imgs: files.append(random.choice(imgs))

# Add test image
files.insert(0, root / 'test_maccup.png')

n = len(files)
fig, axes = plt.subplots(n, 2, figsize=(10, 3*n))
print(f"Processing {n} images...")

for row, fp in enumerate(files):
    img = cv2.imread(str(fp))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    h,w = img.shape[:2]; s=256/max(h,w)
    img = cv2.resize(img, (int(w*s), int(h*s)))
    c = process(img)

    print(f"  {fp.stem}: cu mean={c.mean():.4f} max={c.max():.4f} median={np.median(c):.4f}")

    axes[row,0].imshow((np.clip(img,0,1)*255).astype(np.uint8))
    axes[row,0].set_title(fp.stem, fontsize=9); axes[row,0].axis('off')

    vm = max(c.max(), 0.05)
    axes[row,1].imshow(c, cmap='hot', vmin=0, vmax=vm)
    axes[row,1].set_title(f'Cu [vmax={vm:.2f}]\nmean={c.mean():.3f} max={c.max():.3f}', fontsize=9)
    axes[row,1].axis('off')

plt.suptitle('Cu (roughness) — edge-suppressed local variance', fontsize=12, fontweight='bold')
plt.tight_layout()
out = OUT / 'test_cu_multi.png'
fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out}")
