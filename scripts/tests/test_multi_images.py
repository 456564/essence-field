"""Multi-image dong/jing test — any image or directory"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, cv2, argparse
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, jing

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process_one(img_path, max_size=512):
    """Process single image, return (img_rgb, dong_map, jing_map, stats)"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = dong(x)[0,0].cpu().numpy()
        j = jing(x)[0,0].cpu().numpy()
    stats = {
        'name': img_path.stem, 'shape': img.shape[:2],
        'd_mean': d.mean(), 'd_max': d.max(), 'd_median': np.median(d),
        'j_mean': j.mean(), 'j_median': np.median(j),
    }
    return img, d, j, stats

def render_grid(results, out_name='multi_test.png'):
    """Render all results in a grid"""
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for row, (img, d_map, j_map, st) in enumerate(results):
        img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)
        axes[row,0].imshow(img_u8)
        axes[row,0].set_title(f'{st["name"]}', fontsize=9)
        axes[row,0].axis('off')
        vd = max(d_map.max(), 0.05)
        axes[row,1].imshow(d_map, cmap='hot', vmin=0, vmax=vd)
        axes[row,1].set_title(f'Dong [vmax={vd:.2f}]\nmean={st["d_mean"]:.3f} max={st["d_max"]:.3f}', fontsize=9)
        axes[row,1].axis('off')
        axes[row,2].imshow(j_map, cmap='Blues', vmin=0, vmax=1)
        axes[row,2].set_title(f'Jing\nmean={st["j_mean"]:.3f}', fontsize=9)
        axes[row,2].axis('off')
        # Overlay
        ds = np.clip(d_map/max(d_map.max(),0.01), 0, 1)
        ov = np.stack([ds, np.zeros_like(d_map), j_map], axis=-1)
        axes[row,3].imshow(img_u8)
        axes[row,3].imshow(ov, alpha=0.5)
        axes[row,3].set_title('Overlay', fontsize=9)
        axes[row,3].axis('off')
    plt.suptitle(f'Dong/Jing — {n} images', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = OUT / out_name
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out_path}')
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', help='Image files or directories')
    parser.add_argument('--n', type=int, default=8, help='Max images (default 8)')
    args = parser.parse_args()

    # Find images
    root = Path(__file__).resolve().parent.parent.parent
    if args.paths:
        sources = [Path(p) for p in args.paths]
    else:
        # Default: test images + a few from caltech101
        sources = [root / 'test_maccup.png']
        caltech = root / 'data' / 'caltech101' / '101_ObjectCategories'
        if caltech.exists():
            import random
            cats = [d for d in caltech.iterdir() if d.is_dir() and d.name != 'BACKGROUND_Google']
            for cat in random.sample(cats, min(5, len(cats))):
                imgs = list(cat.glob('*.jpg'))
                if imgs:
                    sources.append(random.choice(imgs))

    # Collect images
    img_files = []
    for src in sources:
        if not src.exists():
            print(f'Not found: {src}')
            continue
        if src.is_dir():
            img_files.extend(sorted(src.glob('*.png')) + sorted(src.glob('*.jpg')) + sorted(src.glob('*.jpeg')))
        elif src.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']:
            img_files.append(src)

    # Limit
    import random
    if len(img_files) > args.n:
        img_files = random.sample(img_files, args.n)
    print(f'Processing {len(img_files)} images...')

    results = []
    for fp in img_files:
        print(f'  {fp.name}...', end=' ', flush=True)
        r = process_one(fp)
        if r:
            results.append(r)
            st = r[3]
            print(f'dong max={st["d_max"]:.3f} mean={st["d_mean"]:.3f}')
        else:
            print('FAILED')

    if results:
        render_grid(results, 'multi_test.png')
    else:
        print('No images processed.')

if __name__ == '__main__':
    main()
