"""Gang operator demo: dong/jing/gang comparison + real photos"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, jing, gang

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process(img):
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = dong(t)[0,0].cpu().numpy()
        j = jing(t)[0,0].cpu().numpy()
        g = gang(t)[0,0].cpu().numpy()
    return d, j, g

S = 128
imgs = {}

# 1. Sharp edge
img = np.zeros((S,S,3), dtype=np.float32)
img[:,:S//2]=[1,0,0]; img[:,S//2:]=[0,0,1]
imgs['Sharp Edge'] = img

# 2. Soft (Gaussian blurred edge)
imgs['Soft Edge'] = cv2.GaussianBlur(img, (5,5), 3)

# 3. Texture
y,x = np.mgrid[:S,:S]; cb=((y//8+x//8)%2).astype(np.float32)
imgs['Texture'] = np.stack([cb*0.8+0.1,cb*0.3+0.2,cb*0.9],axis=-1)

# 4. Soft gradient
ramp=np.linspace(0,1,S).reshape(1,S,1).astype(np.float32)
img_g=np.zeros((S,S,3),dtype=np.float32); img_g[:]=ramp*[0,1,0]
imgs['Soft Gradient'] = img_g

# 5. Real photo
img_path = Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'
if img_path.exists():
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    imgs['Real Photo'] = cv2.resize(img, (S,S))

n = len(imgs)
fig, axes = plt.subplots(n, 4, figsize=(16, 3*n))

for row, (name, img) in enumerate(imgs.items()):
    d, j, g = process(img)
    img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)

    # Col 0: original
    axes[row,0].imshow(img_u8)
    axes[row,0].set_title(f'{name}', fontsize=10); axes[row,0].axis('off')

    # Col 1: dong — fixed vmax=1.0, true absolute scale
    axes[row,1].imshow(d, cmap='hot', vmin=0, vmax=0.5)
    axes[row,1].set_title(f'Dong (change)\nmean={d.mean():.3f} max={d.max():.3f}', fontsize=9)
    axes[row,1].axis('off')

    # Col 2: jing
    axes[row,2].imshow(j, cmap='Blues', vmin=0, vmax=1.0)
    axes[row,2].set_title(f'Jing (still)\nmean={j.mean():.3f}', fontsize=9)
    axes[row,2].axis('off')

    # Col 3: gang — same vmax=1.0 as dong, shows true difference
    axes[row,3].imshow(g, cmap='hot', vmin=0, vmax=0.5)
    axes[row,3].set_title(f'Gang (hard edge)\nmean={g.mean():.3f} max={g.max():.3f}', fontsize=9)
    axes[row,3].axis('off')

plt.suptitle('Dong → Jing → Gang — Physical Operator Pipeline', fontsize=14, fontweight='bold')
plt.tight_layout()
out = OUT / 'demo_gang.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')

# Bonus: real photo detail with overlay
if 'Real Photo' in imgs:
    img = imgs['Real Photo']
    d, j, g = process(img)
    img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_u8); axes[0].set_title('Original'); axes[0].axis('off')

    axes[1].imshow(d, cmap='hot', vmin=0, vmax=0.5)
    axes[1].set_title(f'Dong (all changes)\n{d.mean():.3f} / {d.max():.3f}'); axes[1].axis('off')

    axes[2].imshow(g, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title(f'Gang (hard edges only)\n{g.mean():.3f} / {g.max():.3f}'); axes[2].axis('off')

    # Overlay: dong=green(weak), gang=red(strong)
    ds=np.clip(d/max(d.max(),0.01),0,1); gs=np.clip(g/max(g.max(),0.01),0,1)
    ov=np.stack([gs, ds*0.5, np.zeros_like(d)], axis=-1)
    axes[3].imshow(img_u8); axes[3].imshow(ov, alpha=0.5)
    axes[3].set_title('Overlay: Red=Gang Green=Dong'); axes[3].axis('off')

    plt.tight_layout()
    out2 = OUT / 'demo_gang_real.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out2}')

print('\nDone. Open test_output/demo_gang*.png')
