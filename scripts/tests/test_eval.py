"""Evaluate trained model: zero-shot clustering after training"""
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
    img = cv2.imread(str(img_path))
    h,w = img.shape[:2]; scale = max_size/max(h,w)
    img_disp = cv2.resize(img, (int(w*scale), int(h*scale)))
    rois = []
    for label in ['Object','Background']:
        r = cv2.selectROI(label, img_disp, showCrosshair=True)
        cv2.destroyAllWindows()
        if r[2]==0 or r[3]==0:
            rois.append(None)
        else:
            x,y,w_r,h_r = [int(v/scale) for v in r]
            rois.append((y, y+h_r, x, x+w_r))
    return rois

def test_image(img_path, pipe, obj_rect, bg_rect):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    H,W = img.shape[:2]
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        field = pipe(x)
        raw = pipe.operator_layer(x)

    F = field[0].reshape(field.shape[1], -1).T  # [N, C]
    R = raw[0].reshape(raw.shape[1], -1).T     # [N, 5]
    Fn = F / (F.norm(dim=1, keepdim=True)+1e-8)
    Rn = R / (R.norm(dim=1, keepdim=True)+1e-8)

    def get_roi(vecs, rect):
        y1,y2,x1,x2 = [max(0,int(v)) for v in rect]
        y2=min(H,y2); x2=min(W,x2)
        mask = np.zeros((H,W),bool); mask[y1:y2,x1:x2]=True
        return vecs[np.where(mask.ravel())[0]].cpu().numpy()

    obj_r = get_roi(Rn, obj_rect); bg_r = get_roi(Rn, bg_rect)
    obj_f = get_roi(Fn, obj_rect); bg_f = get_roi(Fn, bg_rect)

    def analyze(o,b,label):
        n=min(500,len(o),len(b))
        o=o[np.random.choice(len(o),n,replace=False)]
        b=b[np.random.choice(len(b),n,replace=False)]
        om,bm=o.mean(axis=0),b.mean(axis=0)
        intra_o=(o*om).sum(axis=1).mean()
        intra_b=(b*bm).sum(axis=1).mean()
        inter=(om*bm).sum()
        sep=min(intra_o,intra_b)-inter
        print(f"  {label}: intra_obj={intra_o:.4f} intra_bg={intra_b:.4f} inter={inter:.4f} sep={sep:.4f} {'OK' if sep>0.05 else 'WEAK'}")
        return sep

    print(f"\n{'='*60}")
    print(f"Eval: {img_path.name}  ({H}x{W})")
    print(f"  Obj: {obj_rect} ({len(obj_r)}px)  Bg: {bg_rect} ({len(bg_r)}px)")
    print(f"  Field dim: {field.shape[1]}")

    sr = analyze(obj_r, bg_r, f"Raw-{raw.shape[1]}op")
    sf = analyze(obj_f, bg_f, f"{field.shape[1]}d-field")

    # Compare with untrained
    print(f"\n  Reference (untrained): sep ≈ 0.0")
    print(f"  Trained improvement: raw +{sr:.4f}, field +{sf:.4f}")

    # Visualize — 2×4 grid
    img_u8 = (np.clip(img,0,1)*255).astype(np.uint8)
    fig,axes = plt.subplots(2,4,figsize=(22,9)); axes=axes.ravel()
    def dr(ax,r,c):
        y1,y2,x1,x2=r; ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=c,lw=2))
    axes[0].imshow(img_u8); axes[0].set_title(img_path.name); axes[0].axis('off')
    dr(axes[0],obj_rect,'lime'); dr(axes[0],bg_rect,'red')

    norms=F.norm(dim=1).reshape(H,W).cpu().numpy()
    axes[1].imshow(norms,cmap='hot',vmin=0,vmax=max(norms.max(),0.01))
    axes[1].set_title(f'Field Norm\nmean={norms.mean():.3f}'); axes[1].axis('off')
    dr(axes[1],obj_rect,'lime'); dr(axes[1],bg_rect,'red')

    for i,n in enumerate(['dong','gang','cu','ju','dist']):
        rn=raw[0,i].cpu().numpy()
        axes[2+i].imshow(rn,cmap='hot',vmin=0,vmax=max(rn.max(),0.05))
        axes[2+i].set_title(f'{n}\nmean={rn.mean():.3f}'); axes[2+i].axis('off')
        dr(axes[2+i],obj_rect,'lime'); dr(axes[2+i],bg_rect,'red')

    # Extra cell: field direction similarity
    Fn_img=Fn.reshape(H,W,-1).cpu().numpy()
    cy,cx=H//2,W//2; cv=Fn_img[cy,cx]
    sim=(Fn_img*cv).sum(axis=-1)
    axes[7].imshow(img_u8,alpha=0.5); axes[7].imshow(sim,cmap='RdYlGn',vmin=-0.5,vmax=1,alpha=0.6)
    axes[7].set_title(f'Dir sim to center\n({cx},{cy})'); axes[7].axis('off')

    plt.suptitle(f'Eval: {img_path.name}',fontsize=12,fontweight='bold')
    plt.tight_layout()
    out=OUT/f'eval_{img_path.stem}.png'
    fig.savefig(out,dpi=120,bbox_inches='tight',facecolor='white')
    plt.close()
    print(f"  Saved: {out}")
    return sr, sf

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('image',help='Image path')
    parser.add_argument('--ckpt',help='Trained checkpoint')
    parser.add_argument('--obj',nargs=4,type=int,metavar=('Y1','Y2','X1','X2'))
    parser.add_argument('--bg',nargs=4,type=int,metavar=('Y1','Y2','X1','X2'))
    args=parser.parse_args()

    img_path=Path(args.image)
    if not img_path.exists():
        print(f"Not found: {img_path}"); sys.exit(1)

    pipe=PhysicalPipeline().to(DEVICE).eval()
    if args.ckpt:
        ckpt=torch.load(args.ckpt,map_location=DEVICE)
        pipe.load_state_dict(ckpt.get('pipe', ckpt.get('model', ckpt)))
        print(f"Loaded: {args.ckpt}")
    else:
        print("WARNING: no checkpoint, using untrained weights")

    if args.obj and args.bg:
        obj_r, bg_r = tuple(args.obj), tuple(args.bg)
    else:
        rois=select_rois(img_path)
        obj_r, bg_r = rois
        if obj_r is None or bg_r is None:
            print("Need both ROIs"); sys.exit(1)

    test_image(img_path, pipe, obj_r, bg_r)
