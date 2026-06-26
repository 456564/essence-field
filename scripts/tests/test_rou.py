"""Rou (柔=soft texture/permeable boundary) operator tests"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, gang, rou

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

pn=0; fn=0
def check(c,n,d=""):
    global pn,fn
    if c: pn+=1; print(f"  PASS: {n}")
    else: fn+=1; print(f"  FAIL: {n} -- {d}")

def t(a):
    return torch.from_numpy(a.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)

# ── Test 1: rou ≈ dong when gang ≈ 0 ──
def test_rou_eq_dong_when_flat():
    print("\n=== Test 1: rou ≈ dong when no hard edges ===")
    S=64
    ramp=np.linspace(0,1,S).reshape(1,S,1).astype(np.float32)
    img=np.zeros((S,S,3),dtype=np.float32); img[:]=ramp*[0,1,0]
    d=dong(t(img))[0,0].cpu().numpy(); r=rou(t(img))[0,0].cpu().numpy()
    diff=np.abs(d-r).mean()
    print(f"  Soft gradient: |dong-rou| mean={diff:.6f}")
    check(diff<0.001, "soft gradient: rou≈dong", f"diff={diff:.6f}")

    # Uniform
    img2=np.full((S,S,3),0.5,dtype=np.float32)
    d2=dong(t(img2))[0,0].cpu().numpy(); r2=rou(t(img2))[0,0].cpu().numpy()
    check(r2.max()<0.01, "uniform: rou≈0", f"max={r2.max():.6f}")

# ── Test 2: rou << dong at sharp edges ──
def test_rou_low_at_sharp_edges():
    print("\n=== Test 2: rou low at sharp edges ===")
    S=128
    img=np.zeros((S,S,3),dtype=np.float32)
    img[:,:S//2]=[1,0,0]; img[:,S//2:]=[0,0,1]
    d=dong(t(img))[0,0].cpu().numpy(); g=gang(t(img))[0,0].cpu().numpy()
    r=rou(t(img))[0,0].cpu().numpy()
    edge_col=S//2
    d_edge=d[64,edge_col]; g_edge=g[64,edge_col]; r_edge=r[64,edge_col]
    print(f"  Sharp edge pixel: dong={d_edge:.4f} gang={g_edge:.4f} rou={r_edge:.4f}")
    check(g_edge>r_edge, "sharp edge: gang>rou", f"gang={g_edge:.4f} rou={r_edge:.4f}")
    check(r_edge<d_edge*0.5, "sharp edge: rou<<dong", f"rou={r_edge:.4f} dong={d_edge:.4f}")

# ── Test 3: soft edge has rou > gang ──
def test_rou_high_at_soft_edge():
    print("\n=== Test 3: rou dominates soft transitions ===")
    S=128
    img=np.zeros((S,S,3),dtype=np.float32)
    img[:,:S//2]=[1,0,0]; img[:,S//2:]=[0,0,1]
    img_soft=cv2.GaussianBlur(img,(5,5),3)
    d=dong(t(img_soft))[0,0].cpu().numpy()
    g=gang(t(img_soft))[0,0].cpu().numpy()
    r=rou(t(img_soft))[0,0].cpu().numpy()
    mid=S//2
    print(f"  Soft edge center: dong={d[64,mid]:.4f} gang={g[64,mid]:.4f} rou={r[64,mid]:.4f}")
    check(r[64,mid]>g[64,mid], "soft edge: rou>gang",
          f"rou={r[64,mid]:.4f} gang={g[64,mid]:.4f}")

# ── Test 4: donkey+gang+rou ≈ dong (invariant) ──
def test_conservation():
    print("\n=== Test 4: rou+gang ≈ dong ===")
    S=64
    # Mix of sharp and soft
    img=np.zeros((S,S,3),dtype=np.float32)
    img[:,:S//3]=[1,0,0]; img[:,2*S//3:]=[0,0,1]
    img_soft=cv2.GaussianBlur(img,(5,5),3)
    d=dong(t(img_soft))[0,0].cpu().numpy()
    g=gang(t(img_soft))[0,0].cpu().numpy()
    r=rou(t(img_soft))[0,0].cpu().numpy()
    diff=np.abs(d-(g+r)).mean()
    print(f"  |dong-(gang+rou)| mean={diff:.6f} max={np.abs(d-(g+r)).max():.6f}")
    check(diff<0.02, f"dong≈gang+rou", f"diff={diff:.6f}")

# ── Test 5: real photo ──
def test_real():
    print("\n=== Test 5: Real Photo ===")
    img_path=Path(__file__).resolve().parent.parent.parent/'test_maccup.png'
    if not img_path.exists(): print("  SKIP"); return
    img=cv2.imread(str(img_path))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x=t(img)
    d=dong(x)[0,0].cpu().numpy(); g=gang(x)[0,0].cpu().numpy()
    r=rou(x)[0,0].cpu().numpy(); j=1/(1+d*10)
    print(f"  dong: mean={d.mean():.4f} max={d.max():.4f}")
    print(f"  gang: mean={g.mean():.4f} max={g.max():.4f}")
    print(f"  rou:  mean={r.mean():.4f} max={r.max():.4f}")
    print(f"  gang+dong+rou pixels >0.01: gang={int((g>0.01).sum())} rou={int((r>0.01).sum())}")
    check(r.mean()>0, "rou non-zero", f"mean={r.mean():.6f}")

    # Visualize
    img_u8=(img*255).astype(np.uint8)
    fig,axes=plt.subplots(1,5,figsize=(22,4))
    axes[0].imshow(img_u8);axes[0].set_title('Original');axes[0].axis('off')
    axes[1].imshow(d,cmap='hot',vmin=0,vmax=0.5)
    axes[1].set_title(f'Dong\nmean={d.mean():.3f}');axes[1].axis('off')
    axes[2].imshow(g,cmap='hot',vmin=0,vmax=0.5)
    axes[2].set_title(f'Gang (hard)\nmean={g.mean():.3f}');axes[2].axis('off')
    axes[3].imshow(r,cmap='hot',vmin=0,vmax=0.5)
    axes[3].set_title(f'Rou (soft)\nmean={r.mean():.3f}');axes[3].axis('off')
    # Overlay: red=gang, green=rou
    gs=np.clip(g/max(g.max(),0.01),0,1); rs=np.clip(r/max(r.max(),0.01),0,1)
    ov=np.stack([gs,rs,np.zeros_like(g)],axis=-1)
    axes[4].imshow(img_u8);axes[4].imshow(ov,alpha=0.5)
    axes[4].set_title('Red=Gang Green=Rou');axes[4].axis('off')
    plt.tight_layout()
    out=OUT/'test_rou_real.png'
    fig.savefig(out,dpi=120,bbox_inches='tight',facecolor='white')
    plt.close()
    print(f"  Saved: {out}")
    print("  PASS: real")

if __name__=='__main__':
    print(f"Device: {DEVICE}")
    test_rou_eq_dong_when_flat()
    test_rou_low_at_sharp_edges()
    test_rou_high_at_soft_edge()
    test_conservation()
    test_real()
    print(f"\n{'='*50}")
    print(f"Results: {pn} PASS, {fn} FAIL")
    print("ALL TESTS PASSED" if fn==0 else "SOME FAILED")
