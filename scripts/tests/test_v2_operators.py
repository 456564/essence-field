"""V2 operators test: dong/gang/cu/ju on synthetic + real"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, gang, cu, ju

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

# ── Test 1: dong/gang unchanged ──
def test_dong_gang():
    print("\n=== Test 1: Dong/Gang regression ===")
    S=128
    img=np.zeros((S,S,3),dtype=np.float32); img[:,:S//2]=[1,0,0]; img[:,S//2:]=[0,0,1]
    d=dong(t(img))[0,0].cpu().numpy(); g=gang(t(img))[0,0].cpu().numpy()
    check(abs(d.max()-0.94)<0.02, "dong peak ~0.94", f"{d.max():.4f}")
    check(g.max()>0.3, "gang detects edge", f"{g.max():.4f}")
    # Soft edge: gang suppressed
    soft=cv2.GaussianBlur(img,(5,5),3)
    d2=dong(t(soft))[0,0].cpu().numpy(); g2=gang(t(soft))[0,0].cpu().numpy()
    check(g2.max()<g.max()*0.7, "gang lower on soft edge", f"sharp={g.max():.3f} soft={g2.max():.3f}")

# ── Test 2: cu — texture vs smooth ──
def test_cu():
    print("\n=== Test 2: Cu (roughness) ===")
    S=64
    # Smooth
    img_s=np.full((S,S,3),0.5,dtype=np.float32)
    cu_s=cu(t(img_s))[0,0].cpu().numpy()
    check(cu_s.max()<0.01, "smooth surface cu≈0", f"max={cu_s.max():.6f}")
    # Textured: random blocks
    np.random.seed(42)
    blocks=np.random.rand(S//8,S//8,3).astype(np.float32)
    img_t=cv2.resize(blocks,(S,S),interpolation=cv2.INTER_NEAREST)
    cu_t=cu(t(img_t))[0,0].cpu().numpy()
    check(cu_t.mean()>cu_s.mean()*5, "textured >> smooth cu",
          f"tex={cu_t.mean():.4f} smooth={cu_s.mean():.4f}")
    # Gradient: should have low cu (smooth transition, not rough)
    ramp=np.linspace(0,1,S).reshape(1,S,1).astype(np.float32)
    img_g=np.zeros((S,S,3),dtype=np.float32); img_g[:]=ramp*[0,1,0]
    cu_g=cu(t(img_g))[0,0].cpu().numpy()
    interior=cu_g[2:-2,2:-2]
    check(interior.mean()<0.05, "gradient cu low (smooth transition)", f"mean={interior.mean():.4f}")

# ── Test 3: ju — enclosed vs open (check near edge, 31x31 window local) ──
def test_ju():
    print("\n=== Test 3: Ju (enclosure) ===")
    S=128
    # Small enclosed square: edge at 32, center at 48 → 16px from edge → in 31x31 window
    img=np.ones((S,S,3),dtype=np.float32)*0.5
    r=32; img[r:S-r,r:S-r]=[0.9,0.9,0.9]  # 64x64 white square
    j=ju(t(img))[0,0].cpu().numpy()
    # Near edge of square: should see gang in 31x31 window
    near_edge=j[r+8,r+8]  # 8px inside the square edge
    far_corner=j[4,4]     # far from any edges
    print(f"  Square: ju_near_edge={near_edge:.4f} ju_far={far_corner:.4f}")
    check(near_edge>far_corner*1.5, "ju near enclosure edge > far corner",
          f"near={near_edge:.4f} far={far_corner:.4f}")

    # 31x31 kernel can't see edges >15px away — this is correct local behavior
    center=j[S//2,S//2]
    print(f"  Square center (32px from edges): ju={center:.4f} (expected ~0, beyond 31x31 window)")

# ── Test 4: cu & dong independence ──
def test_cu_dong_independent():
    print("\n=== Test 4: Cu/Dong independence ===")
    S=64
    # Sharp edge: high dong, low cu
    img_e=np.zeros((S,S,3),dtype=np.float32); img_e[:,:S//2]=[1,0,0]; img_e[:,S//2:]=[0,0,1]
    d_e=dong(t(img_e))[0,0].cpu().numpy(); c_e=cu(t(img_e))[0,0].cpu().numpy()
    # Texture: medium dong, high cu
    np.random.seed(7)
    b=np.random.rand(S//4,S//4,3).astype(np.float32)
    img_t=cv2.resize(b,(S,S),interpolation=cv2.INTER_NEAREST)
    d_t=dong(t(img_t))[0,0].cpu().numpy(); c_t=cu(t(img_t))[0,0].cpu().numpy()
    print(f"  Edge:    dong_max={d_e.max():.3f} cu_max={c_e.max():.3f}")
    print(f"  Texture: dong_max={d_t.max():.3f} cu_max={c_t.max():.3f}")
    # Key: cu separates them — texture >> edge in roughness
    check(c_t.max()>c_e.max()*1.3, "cu: texture >> edge roughness",
          f"tex_cu={c_t.max():.3f} edge_cu={c_e.max():.3f}")
    print(f"  (dong similar — both have sharp individual transitions)")

# ── Test 5: real photo visualization ──
def test_real():
    print("\n=== Test 5: Real Photo ===")
    img_path=Path(__file__).resolve().parent.parent.parent/'test_maccup.png'
    if not img_path.exists(): print("  SKIP"); return
    img=cv2.imread(str(img_path))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x=t(img)
    d=dong(x)[0,0].cpu().numpy(); g=gang(x)[0,0].cpu().numpy()
    c=cu(x)[0,0].cpu().numpy(); j=ju(x)[0,0].cpu().numpy()

    print(f"  dong: mean={d.mean():.4f} max={d.max():.4f}")
    print(f"  gang: mean={g.mean():.4f} max={g.max():.4f}")
    print(f"  cu:   mean={c.mean():.4f} max={c.max():.4f}")
    print(f"  ju:   mean={j.mean():.4f} max={j.max():.4f}")

    # Visualize
    img_u8=(img*255).astype(np.uint8)
    fig,axes=plt.subplots(1,5,figsize=(22,4))
    axes[0].imshow(img_u8);axes[0].set_title('Original');axes[0].axis('off')
    axes[1].imshow(d,cmap='hot',vmin=0,vmax=0.5)
    axes[1].set_title(f'Dong (gradient)\nmean={d.mean():.3f} max={d.max():.3f}');axes[1].axis('off')
    axes[2].imshow(g,cmap='hot',vmin=0,vmax=0.5)
    axes[2].set_title(f'Gang (ridge)\nmean={g.mean():.3f} max={g.max():.3f}');axes[2].axis('off')
    axes[3].imshow(c,cmap='hot',vmin=0,vmax=0.05)
    axes[3].set_title(f'Cu (roughness)\nmean={c.mean():.3f} max={c.max():.3f}');axes[3].axis('off')
    axes[4].imshow(j,cmap='hot',vmin=0,vmax=0.5)
    axes[4].set_title(f'Ju (enclosure)\nmean={j.mean():.3f} max={j.max():.3f}');axes[4].axis('off')
    plt.tight_layout()
    out=OUT/'test_v2_real.png'
    fig.savefig(out,dpi=120,bbox_inches='tight',facecolor='white')
    plt.close()
    print(f"  Saved: {out}")
    print("  PASS: real")

if __name__=='__main__':
    print(f"Device: {DEVICE}")
    test_dong_gang()
    test_cu()
    test_ju()
    test_cu_dong_independent()
    test_real()
    print(f"\n{'='*50}")
    print(f"Results: {pn} PASS, {fn} FAIL")
    print("ALL TESTS PASSED" if fn==0 else "SOME FAILED")
