"""Single operator tests: one op at a time, numeric + visual"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2, argparse
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, gang, cu, rou, ju, dist, yang, yin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def process_one(img, op_fn, op_name):
    t = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = op_fn(t)[0,0].cpu().numpy()
    return out

S = 128

# ── Synthetic test patterns ──
def make_synthetic():
    tests = {}
    # 1. Sharp red/blue edge
    img = np.zeros((S,S,3), dtype=np.float32)
    img[:,:S//2]=[1,0,0]; img[:,S//2:]=[0,0,1]
    tests['Sharp Edge'] = img
    # 2. Soft blurred edge
    tests['Soft Edge'] = cv2.GaussianBlur(img, (5,5), 3)
    # 3. Texture checkerboard
    y,x=np.mgrid[:S,:S]; cb=((y//8+x//8)%2).astype(np.float32)
    tests['Texture']=np.stack([cb*0.8+0.1,cb*0.3+0.2,cb*0.9],axis=-1)
    # 4. Soft gradient
    ramp=np.linspace(0,1,S).reshape(1,S,1).astype(np.float32)
    img_g=np.zeros((S,S,3),dtype=np.float32); img_g[:]=ramp*[0,1,0]
    tests['Soft Gradient']=img_g
    # 5. Uniform gray
    tests['Uniform']=np.full((S,S,3),0.5,dtype=np.float32)
    return tests

def test_op(op_fn, op_name):
    print(f"\n{'='*60}")
    print(f"Testing: {op_name}")
    print(f"{'='*60}")

    # Synthetic tests
    syn = make_synthetic()
    print(f"\n{'Scene':<18} {'Min':>8} {'Max':>8} {'Mean':>8} {'Median':>8}")
    print("-"*54)
    for name, img in syn.items():
        out = process_one(img, op_fn, op_name)
        print(f"{name:<18} {out.min():8.4f} {out.max():8.4f} {out.mean():8.4f} {np.median(out):8.4f}")

    # Real photo
    img_path = Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'
    if img_path.exists():
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        out = process_one(img, op_fn, op_name)
        print(f"{'Real Photo':<18} {out.min():8.4f} {out.max():8.4f} {out.mean():8.4f} {np.median(out):8.4f}")
        real_out = out
        real_img = img
    else:
        real_out = None
        real_img = None

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    idx = 0
    for name, img in syn.items():
        out = process_one(img, op_fn, op_name)
        img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)
        vmax = max(out.max(), 0.05)
        axes[idx//3, idx%3].imshow(img_u8, alpha=0.5)
        axes[idx//3, idx%3].imshow(out, cmap='hot', vmin=0, vmax=vmax, alpha=0.7)
        axes[idx//3, idx%3].set_title(f'{name}\nmin={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}')
        axes[idx//3, idx%3].axis('off')
        idx += 1

    if real_out is not None:
        vmax = max(real_out.max(), 0.05)
        axes[1,2].imshow((np.clip(real_img,0,1)*255).astype(np.uint8), alpha=0.5)
        axes[1,2].imshow(real_out, cmap='hot', vmin=0, vmax=vmax, alpha=0.7)
        axes[1,2].set_title(f'Real Photo\nmin={real_out.min():.3f} max={real_out.max():.3f} mean={real_out.mean():.3f}')
        axes[1,2].axis('off')

    out_path = OUT / f'test_single_{op_name}.png'
    plt.suptitle(f'Operator: {op_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('op', nargs='?', choices=['dong','gang','cu','rou','ju','dist','yang','yin','all'],
                        default='all', help='Operator to test')
    args = parser.parse_args()

    ops = {'dong': dong, 'gang': gang, 'cu': cu, 'rou': rou, 'ju': ju, 'dist': dist, 'yang': yang, 'yin': yin}
    if args.op == 'all':
        for name, fn in ops.items():
            test_op(fn, name)
    else:
        test_op(ops[args.op], args.op)
