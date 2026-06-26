"""Gang (刚=hard boundary) operator tests"""
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

pass_n = 0; fail_n = 0
def check(cond, name, detail=""):
    global pass_n, fail_n
    if cond: pass_n += 1; print(f"  PASS: {name}")
    else: fail_n += 1; print(f"  FAIL: {name} -- {detail}")

def t(arr):
    return torch.from_numpy(arr.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)

# ── Test 1: sharp edge vs soft transition ──
def test_sharp_vs_soft():
    print("\n=== Test 1: Sharp vs Soft ===")
    S = 128
    # Sharp: 1px red/blue edge
    img_s = np.zeros((S,S,3), dtype=np.float32)
    img_s[:,:S//2]=[1,0,0]; img_s[:,S//2:]=[0,0,1]
    g_s = gang(t(img_s))[0,0].cpu().numpy()
    print(f"  Sharp edge: gang max={g_s.max():.4f} mean={g_s.mean():.4f}")

    # Soft: 5px transition via box blur
    img_soft = cv2.GaussianBlur(img_s, (5,5), 3)  # softens the edge
    g_soft = gang(t(img_soft))[0,0].cpu().numpy()
    print(f"  Soft edge:  gang max={g_soft.max():.4f} mean={g_soft.mean():.4f}")

    ratio = g_s.max() / max(g_soft.max(), 1e-8)
    print(f"  Sharp/Soft ratio: {ratio:.1f}x")
    check(ratio > 2.0, "sharp edge gang >> soft edge", f"ratio={ratio:.1f}")

# ── Test 2: texture edges ARE hard (individual pixel transitions are sharp) ──
def test_texture():
    print("\n=== Test 2: Texture (checkerboard edges are hard) ===")
    S = 128
    y,x = np.mgrid[:S,:S]
    cb = ((y//8 + x//8) % 2).astype(np.float32)
    img_tex = np.stack([cb*0.8+0.1, cb*0.3+0.2, cb*0.9], axis=-1)
    g_tex = gang(t(img_tex))[0,0].cpu().numpy()
    d_tex = dong(t(img_tex))[0,0].cpu().numpy()
    print(f"  Checkerboard: dong max={d_tex.max():.4f} gang max={g_tex.max():.4f}")
    print(f"    dong mean={d_tex.mean():.4f} gang mean={g_tex.mean():.4f}")
    # Each block edge IS a 1px hard transition → gang should detect it
    check(g_tex.max() > 0.3, "checkerboard edges detected as hard",
          f"gang max={g_tex.max():.4f}")
    # But gang mean should be moderate (not all pixels are edges)
    check(g_tex.mean() < 0.3, "gang sparse in checkerboard (only at edges)",
          f"gang mean={g_tex.mean():.4f}")

# ── Test 3: uniform & gradient → gang ≈ 0 ──
def test_flat_is_zero():
    print("\n=== Test 3: Flat/Gradient → gang≈0 ===")
    S = 64
    # Uniform
    img_u = np.full((S,S,3), 0.5, dtype=np.float32)
    g_u = gang(t(img_u))[0,0].cpu().numpy()
    check(g_u.max() < 0.01, f"uniform gang≈0", f"max={g_u.max():.6f}")

    # Soft gradient
    ramp = np.linspace(0,1,S).reshape(1,S,1).astype(np.float32)
    img_g = np.zeros((S,S,3), dtype=np.float32); img_g[:]=ramp*[0,1,0]
    g_g = gang(t(img_g))[0,0].cpu().numpy()
    # Interor should be near 0 (uniform gradient = no isolated edge)
    interior = g_g[2:-2, 2:-2]
    check(interior.mean() < 0.01, f"soft gradient gang≈0", f"mean={interior.mean():.6f}")

# ── Test 4: real photo ──
def test_real():
    print("\n=== Test 4: Real Photo ===")
    img_path = Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'
    if not img_path.exists():
        print("  SKIP: no image")
        return
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = t(img)
    d = dong(x)[0,0].cpu().numpy()
    g = gang(x)[0,0].cpu().numpy()
    j = jing(x)[0,0].cpu().numpy()
    print(f"  dong: mean={d.mean():.4f} max={d.max():.4f} median={np.median(d):.4f}")
    print(f"  gang: mean={g.mean():.4f} max={g.max():.4f} median={np.median(g):.4f}")
    print(f"  jing: mean={j.mean():.4f} max={j.max():.4f} median={np.median(j):.4f}")
    # Gang should be sparser than dong (fewer pixels active)
    d_active = (d > d.mean()).sum()
    g_active = (g > 0.01).sum()
    print(f"  dong active pixels: {d_active} ({100*d_active/d.size:.1f}%)")
    print(f"  gang active pixels: {g_active} ({100*g_active/g.size:.1f}%)")
    check(g_active < d_active * 0.5, "gang sparser than dong",
          f"gang={g_active} dong={d_active}")

    # Visualize
    img_u8 = (img*255).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(img_u8); axes[0].set_title('Original'); axes[0].axis('off')
    vd = max(d.max(), 0.05)
    axes[1].imshow(d, cmap='hot', vmin=0, vmax=vd)
    axes[1].set_title(f'Dong\nmean={d.mean():.3f} max={d.max():.3f}'); axes[1].axis('off')
    vg = max(g.max(), 0.05)
    axes[2].imshow(g, cmap='hot', vmin=0, vmax=vg)
    axes[2].set_title(f'Gang\nmean={g.mean():.3f} max={g.max():.3f}'); axes[2].axis('off')
    # Overlay: gang=red on original
    gs = np.clip(g/max(g.max(),0.01),0,1)
    ov = np.stack([gs, np.zeros_like(g), np.zeros_like(g)], axis=-1)
    axes[3].imshow(img_u8); axes[3].imshow(ov, alpha=0.6)
    axes[3].set_title('Gang overlay (red)'); axes[3].axis('off')
    plt.tight_layout()
    out = OUT / 'test_gang_real.png'
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")
    print("  PASS: real")

# ── Test 5: gang consistency across scales ──
def test_scale():
    print("\n=== Test 5: Scale Consistency ===")
    for size in [64, 128, 256]:
        img = np.zeros((size,size,3), dtype=np.float32)
        img[:,:size//2]=[1,0,0]; img[:,size//2:]=[0,0,1]
        g = gang(t(img))[0,0].cpu().numpy()
        check(g.max() > 0.5, f"sharp edge {size}x{size} detected",
              f"max={g.max():.4f}")

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    test_sharp_vs_soft()
    test_texture()
    test_flat_is_zero()
    test_real()
    test_scale()
    print(f"\n{'='*50}")
    print(f"Results: {pass_n} PASS, {fail_n} FAIL")
    print("ALL TESTS PASSED" if fail_n==0 else "SOME FAILED")
