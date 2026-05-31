"""Rou operator multi-image test"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2, random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, gang, cu, rou

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process(img):
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = dong(t)[0,0].cpu().numpy()
        g = gang(t)[0,0].cpu().numpy()
        c = cu(t)[0,0].cpu().numpy()
        r = rou(t)[0,0].cpu().numpy()
    return d, g, c, r

root = Path(__file__).resolve().parent.parent.parent
caltech = root / 'data' / 'caltech101' / '101_ObjectCategories'
files = []
if caltech.exists():
    for cat in random.sample([d for d in caltech.iterdir() if d.is_dir() and d.name!='BACKGROUND_Google'], 7):
        imgs = list(cat.glob('*.jpg'))
        if imgs: files.append(random.choice(imgs))
files.insert(0, root / 'test_maccup.png')

n = len(files)
fig, axes = plt.subplots(n, 5, figsize=(22, 3*n))
print(f"Processing {n} images...")

for row, fp in enumerate(files):
    img = cv2.imread(str(fp))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    h,w = img.shape[:2]; s=256/max(h,w)
    img = cv2.resize(img, (int(w*s), int(h*s)))
    d,g,c,r = process(img)

    print(f"  {fp.stem}: d={d.max():.2f} g={g.max():.2f} c={c.max():.2f} r={r.mean():.3f}")

    img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)
    axes[row,0].imshow(img_u8); axes[row,0].set_title(fp.stem, fontsize=8); axes[row,0].axis('off')
    axes[row,1].imshow(d, cmap='hot', vmin=0, vmax=0.5)
    axes[row,1].set_title(f'Dong\nmax={d.max():.2f}', fontsize=8); axes[row,1].axis('off')
    axes[row,2].imshow(g, cmap='hot', vmin=0, vmax=0.5)
    axes[row,2].set_title(f'Gang\nmax={g.max():.2f}', fontsize=8); axes[row,2].axis('off')
    axes[row,3].imshow(c, cmap='hot', vmin=0, vmax=0.1)
    axes[row,3].set_title(f'Cu\nmax={c.max():.2f}', fontsize=8); axes[row,3].axis('off')
    axes[row,4].imshow(r, cmap='hot', vmin=0, vmax=1)
    axes[row,4].set_title(f'Rou\nmean={r.mean():.2f}', fontsize=8); axes[row,4].axis('off')

plt.suptitle('Dong / Gang / Cu / Rou', fontsize=12, fontweight='bold')
plt.tight_layout()
out = OUT / 'test_rou_multi.png'
fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out}")
