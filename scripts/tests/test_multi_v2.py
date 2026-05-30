"""4-op multi-image test — Caltech101 real photos"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2, argparse, random
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, gang, cu, ju

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process(img):
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = dong(t)[0,0].cpu().numpy()
        g = gang(t)[0,0].cpu().numpy()
        c = cu(t)[0,0].cpu().numpy()
        j = ju(t)[0,0].cpu().numpy()
    return d, g, c, j

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--n', type=int, default=6)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    sources = args.paths if args.paths else [str(root / 'data' / 'caltech101' / '101_ObjectCategories')]

    files = []
    for src in sources:
        p = Path(src)
        if not p.exists(): continue
        if p.is_dir():
            files.extend(list(p.rglob('*.jpg')) + list(p.rglob('*.png')))
        elif p.suffix.lower() in ['.png','.jpg','.jpeg']:
            files.append(p)

    if len(files) > args.n:
        files = random.sample(files, args.n)

    n = len(files)
    print(f'Processing {n} images...')
    fig, axes = plt.subplots(n, 5, figsize=(22, 3*n))
    if n == 1: axes = axes.reshape(1,-1)

    for row, fp in enumerate(files):
        print(f'  {fp.name}...', end=' ', flush=True)
        img = cv2.imread(str(fp))
        if img is None:
            print('SKIP'); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        h,w = img.shape[:2]; scale = 256/max(h,w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        d,g,c,j = process(img)
        print(f'd={d.max():.2f} g={g.max():.2f} c={c.max():.2f} j={j.max():.2f}')

        img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)
        axes[row,0].imshow(img_u8); axes[row,0].set_title(fp.stem, fontsize=8); axes[row,0].axis('off')
        axes[row,1].imshow(d, cmap='hot', vmin=0, vmax=0.5)
        axes[row,1].set_title(f'Dong (grad)\nmax={d.max():.2f}', fontsize=8); axes[row,1].axis('off')
        axes[row,2].imshow(g, cmap='hot', vmin=0, vmax=0.5)
        axes[row,2].set_title(f'Gang (ridge)\nmax={g.max():.2f}', fontsize=8); axes[row,2].axis('off')
        axes[row,3].imshow(c, cmap='hot', vmin=0, vmax=0.1)
        axes[row,3].set_title(f'Cu (rough)\nmax={c.max():.2f}', fontsize=8); axes[row,3].axis('off')
        axes[row,4].imshow(j, cmap='hot', vmin=0, vmax=0.5)
        axes[row,4].set_title(f'Ju (enclose)\nmax={j.max():.2f}', fontsize=8); axes[row,4].axis('off')

    plt.suptitle('V2: Dong / Gang / Cu / Ju', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = OUT / 'multi_v2_real.png'
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved: {out}')

if __name__ == '__main__':
    main()
