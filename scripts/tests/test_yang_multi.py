"""Yang multi-image test — 8 columns: orig|dong|gang|ju|dist|yang|yin|overlay"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2, random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, gang, ju, dist, yang, yin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process(img):
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = dong(t)[0,0].cpu().numpy()
        g = gang(t)[0,0].cpu().numpy()
        j = ju(t)[0,0].cpu().numpy()
        dt= dist(t)[0,0].cpu().numpy()
        y = yang(t)[0,0].cpu().numpy()
        yn= yin(t)[0,0].cpu().numpy()
    return d, g, j, dt, y, yn

root = Path(__file__).resolve().parent.parent.parent
caltech = root / 'data' / 'caltech101' / '101_ObjectCategories'
files = []

# Synthetic
S=128; img_s=np.ones((S,S,3),dtype=np.float32)*0.5; img_s[25:103,25:103]=[0.9,0.9,0.9]
files.append(('closed_rect', img_s))

# Real cup
img_cup=cv2.imread(str(root/'test_maccup.png'))
files.append(('cup', cv2.cvtColor(img_cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255))

# Caltech
if caltech.exists():
    cats = [d for d in caltech.iterdir() if d.is_dir() and d.name!='BACKGROUND_Google']
    for cat in random.sample(cats, min(6, len(cats))):
        imgs=list(cat.glob('*.jpg'))
        if imgs:
            p=random.choice(imgs)
            img=cv2.imread(str(p))
            files.append((p.stem, cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255))

n=len(files)
fig, axes = plt.subplots(n, 8, figsize=(28, 3*n))
if n==1: axes=axes.reshape(1,-1)
print(f"Image              dmax  gmax  jmax  j>0.5   yangM  yMean  y>0.5")
print("-"*78)

for row, (name, img) in enumerate(files):
    h,w=img.shape[:2]; s=256/max(h,w)
    img_r=cv2.resize(img, (int(w*s), int(h*s)))
    d,g,j,dt,y,yn = process(img_r)
    j_hi = (j>0.5).mean(); y_hi = (y>0.5).mean()
    print(f"{name:<20} {d.max():.2f}   {g.max():.2f}   {j.max():.2f}   {j_hi:.2f}    {y.max():.2f}    {y.mean():.3f}  {y_hi:.2f}")

    img_u8=(np.clip(img_r,0,1)*255).astype(np.uint8)
    axes[row,0].imshow(img_u8); axes[row,0].set_title(name, fontsize=7); axes[row,0].axis('off')
    axes[row,1].imshow(d, cmap='hot', vmin=0, vmax=0.5)
    axes[row,1].set_title(f'Dong\nmax={d.max():.2f}', fontsize=7); axes[row,1].axis('off')
    axes[row,2].imshow(g, cmap='hot', vmin=0, vmax=0.5)
    axes[row,2].set_title(f'Gang\nmax={g.max():.2f}', fontsize=7); axes[row,2].axis('off')
    axes[row,3].imshow(j, cmap='hot', vmin=0, vmax=1)
    axes[row,3].set_title(f'Ju\nmax={j.max():.2f}', fontsize=7); axes[row,3].axis('off')
    axes[row,4].imshow(dt, cmap='hot', vmin=0, vmax=1)
    axes[row,4].set_title(f'Dist\nmax={dt.max():.2f}', fontsize=7); axes[row,4].axis('off')
    axes[row,5].imshow(y, cmap='hot', vmin=0, vmax=1)
    axes[row,5].set_title(f'Yang (entity)\nmax={y.max():.2f} mean={y.mean():.2f}', fontsize=7); axes[row,5].axis('off')
    axes[row,6].imshow(yn, cmap='Blues', vmin=0, vmax=1)
    axes[row,6].set_title(f'Yin (void)\nmax={yn.max():.2f}', fontsize=7); axes[row,6].axis('off')
    # Overlay: red=yang, blue=yin on original
    ov=np.stack([y, np.zeros_like(y), yn], axis=-1)
    axes[row,7].imshow(img_u8, alpha=0.6); axes[row,7].imshow(ov, alpha=0.4)
    axes[row,7].set_title('Overlay\nred=Yang blue=Yin', fontsize=7); axes[row,7].axis('off')

plt.suptitle('8 Operators: Dong Gang Ju Dist → Yang Yin (Entity/Void)', fontsize=11, fontweight='bold')
plt.tight_layout()
out=OUT/'test_yang_multi.png'
fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out}")
