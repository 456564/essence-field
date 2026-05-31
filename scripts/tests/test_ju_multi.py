"""Ju operator multi-image test"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2, random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, gang, ju

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process(img):
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = dong(t)[0,0].cpu().numpy()
        g = gang(t)[0,0].cpu().numpy()
        j = ju(t)[0,0].cpu().numpy()
    return d, g, j

root = Path(__file__).resolve().parent.parent.parent
caltech = root / 'data' / 'caltech101' / '101_ObjectCategories'
files = []

# Add synthetic closed shape
S=128; img_s=np.ones((S,S,3),dtype=np.float32)*0.5; img_s[25:103,25:103]=[0.9,0.9,0.9]
files.append(('closed_rect', img_s))

# Real cup
img_cup=cv2.imread(str(root/'test_maccup.png'))
img_cup=cv2.cvtColor(img_cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
files.append(('cup', img_cup))

# Caltech samples
if caltech.exists():
    for cat in random.sample([d for d in caltech.iterdir() if d.is_dir() and d.name!='BACKGROUND_Google'], 6):
        imgs=list(cat.glob('*.jpg'))
        if imgs:
            p=random.choice(imgs)
            img=cv2.imread(str(p)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            files.append((p.stem, img))

n=len(files)
fig, axes = plt.subplots(n, 4, figsize=(18, 3*n))
print(f"Processing {n} images...")

for row, (name, img) in enumerate(files):
    h,w=img.shape[:2]; s=256/max(h,w)
    img_r=cv2.resize(img, (int(w*s), int(h*s)))
    d,g,j = process(img_r)

    print(f"  {name}: dmax={d.max():.2f} gmax={g.max():.2f} jmax={j.max():.2f} jmean={j.mean():.3f}")

    img_u8=(np.clip(img_r,0,1)*255).astype(np.uint8)
    axes[row,0].imshow(img_u8); axes[row,0].set_title(name, fontsize=8); axes[row,0].axis('off')
    axes[row,1].imshow(d, cmap='hot', vmin=0, vmax=0.5)
    axes[row,1].set_title(f'Dong\nmax={d.max():.2f}', fontsize=8); axes[row,1].axis('off')
    axes[row,2].imshow(g, cmap='hot', vmin=0, vmax=0.5)
    axes[row,2].set_title(f'Gang\nmax={g.max():.2f}', fontsize=8); axes[row,2].axis('off')
    axes[row,3].imshow(j, cmap='hot', vmin=0, vmax=1)
    axes[row,3].set_title(f'Ju\nmax={j.max():.2f}', fontsize=8); axes[row,3].axis('off')

plt.suptitle('Dong / Gang / Ju — Topological Enclosure', fontsize=12, fontweight='bold')
plt.tight_layout()
out=OUT/'test_ju_multi.png'
fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out}")
