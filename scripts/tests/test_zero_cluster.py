"""Zero-training clustering: interactive ROI selection"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2, argparse
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.pipeline import PhysicalPipeline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)


def select_rois(img_path, max_size=400):
    """Interactive ROI selection with OpenCV"""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    img_disp = cv2.resize(img, (int(w*scale), int(h*scale)))

    rois = []
    for label in ['Object (green)', 'Background (red)']:
        print(f"\n>>> Select {label}: draw rectangle, press ENTER, then press 'c' to confirm <<<")
        r = cv2.selectROI(f'Select {label} — ENTER then c', img_disp, showCrosshair=True)
        cv2.destroyAllWindows()
        if r[2] == 0 or r[3] == 0:
            print(f"  SKIPPED (empty selection)")
            rois.append(None)
        else:
            # Scale back to original size
            x, y, w_r, h_r = [int(v/scale) for v in r]
            rois.append((y, y+h_r, x, x+w_r))
            print(f"  ROI: y=[{y},{y+h_r}] x=[{x},{x+w_r}] ({h_r}x{w_r})")
    return rois


def test_image(img_path, name, obj_rect, bg_rect):
    pipe = PhysicalPipeline().to(DEVICE).eval()
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    H, W = img.shape[:2]
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        field = pipe(x)
        raw = pipe.operator_layer(x)

    F = field[0].reshape(16, -1).T
    R = raw[0].reshape(4, -1).T
    Fn = F / (F.norm(dim=1, keepdim=True) + 1e-8)
    Rn = R / (R.norm(dim=1, keepdim=True) + 1e-8)

    def get_roi(vecs, rect):
        y1,y2,x1,x2 = [max(0,int(v)) for v in rect]
        y2=min(H,y2); x2=min(W,x2)
        mask = np.zeros((H,W), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return vecs[np.where(mask.ravel())[0]].cpu().numpy()

    obj_4 = get_roi(Rn, obj_rect)
    bg_4  = get_roi(Rn, bg_rect)
    obj_16= get_roi(Fn, obj_rect)
    bg_16 = get_roi(Fn, bg_rect)

    def analyze(o, b, label):
        n = min(500, len(o), len(b))
        o = o[np.random.choice(len(o), n, replace=False)]
        b = b[np.random.choice(len(b), n, replace=False)]
        om, bm = o.mean(axis=0), b.mean(axis=0)
        intra_o = (o * om).sum(axis=1).mean()
        intra_b = (b * bm).sum(axis=1).mean()
        inter = (om * bm).sum()
        sep = min(intra_o, intra_b) - inter
        print(f"  {label}: obj_in={intra_o:.4f} bg_in={intra_b:.4f} cross={inter:.4f} sep={sep:.4f} {'OK' if sep>0.05 else 'WEAK'}")
        return sep

    print(f"\n{'='*60}")
    print(f"Test: {name}  ({H}x{W})")
    print(f"  Obj: {obj_rect} ({len(obj_4)}px)  Bg: {bg_rect} ({len(bg_4)}px)")

    sr = analyze(obj_4, bg_4, "Raw-4op")
    s16 = analyze(obj_16, bg_16, "16-dim ")

    # Visualize
    img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    def dr(ax, rect, c):
        y1,y2,x1,x2 = rect
        ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=c,linewidth=2))

    axes[0].imshow(img_u8); axes[0].set_title(name); axes[0].axis('off')
    dr(axes[0], obj_rect, 'lime'); dr(axes[0], bg_rect, 'red')

    norms = F.norm(dim=1).reshape(H,W).cpu().numpy()
    axes[1].imshow(norms, cmap='hot', vmin=0, vmax=max(norms.max(),0.01))
    axes[1].set_title(f'Field Norm\nmean={norms.mean():.3f}'); axes[1].axis('off')
    dr(axes[1], obj_rect, 'lime'); dr(axes[1], bg_rect, 'red')

    for i, name_op in enumerate(['dong','gang','cu','ju']):
        rn = raw[0,i].cpu().numpy()
        axes[2+i].imshow(rn, cmap='hot', vmin=0, vmax=max(rn.max(),0.05))
        axes[2+i].set_title(f'{name_op}\nmean={rn.mean():.3f}'); axes[2+i].axis('off')
        dr(axes[2+i], obj_rect, 'lime'); dr(axes[2+i], bg_rect, 'red')

    plt.suptitle(f'Zero-Training: {name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = OUT / f'zero_cluster_{name}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")
    return sr, s16


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', help='Image path')
    parser.add_argument('--obj', nargs=4, type=int, metavar=('Y1','Y2','X1','X2'), help='Object ROI')
    parser.add_argument('--bg', nargs=4, type=int, metavar=('Y1','Y2','X1','X2'), help='Background ROI')
    args = parser.parse_args()

    if args.image:
        img_path = Path(args.image)
    else:
        img_path = Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'

    if not img_path.exists():
        print(f"Not found: {img_path}")
        sys.exit(1)

    # Get ROIs
    if args.obj and args.bg:
        obj_rect = tuple(args.obj)
        bg_rect = tuple(args.bg)
    else:
        rois = select_rois(img_path)
        obj_rect, bg_rect = rois
        if obj_rect is None or bg_rect is None:
            print("Need both object and background ROI. Use --obj/--bg flags.")
            sys.exit(1)

    test_image(img_path, img_path.stem, obj_rect, bg_rect)
