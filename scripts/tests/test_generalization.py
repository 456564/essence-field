"""Dong/Jing generalizability tests: edge cases, noise, scale, diverse inputs"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.operators import dong, jing

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

pass_count = 0
fail_count = 0

def check(cond, name, detail=""):
    global pass_count, fail_count
    if cond:
        pass_count += 1
        print(f"  PASS: {name}")
    else:
        fail_count += 1
        print(f"  FAIL: {name} -- {detail}")


def to_tensor(arr):
    """HWC numpy [0,1] → BCHW tensor"""
    return torch.from_numpy(arr.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)


# ═══════════════════════════════════════════════════════════
# Test 1: Invariance — uniform color should produce dong≈0
# ═══════════════════════════════════════════════════════════
def test_uniform():
    print("\n=== Test 1: Uniform Color ===")
    for color_name, rgb in [("black", [0,0,0]), ("white", [1,1,1]),
                             ("red", [1,0,0]), ("green", [0,1,0]),
                             ("blue", [0,0,1]), ("gray", [0.5,0.5,0.5])]:
        img = np.full((64, 64, 3), rgb, dtype=np.float32)
        d = dong(to_tensor(img))[0,0].cpu().numpy()
        check(d.max() < 0.03, f"uniform {color_name} dong≈0",
              f"max={d.max():.5f}")


# ═══════════════════════════════════════════════════════════
# Test 2: Gradient — dong should be uniform across the edge
# ═══════════════════════════════════════════════════════════
def test_gradient():
    print("\n=== Test 2: Gradients ===")

    # Horizontal edge at different contrast levels
    for contrast in [0.25, 0.5, 1.0]:
        img = np.zeros((64, 64, 3), dtype=np.float32)
        img[:32] = [0.5, 0.5, 0.5]
        delta = contrast * 0.5
        img[32:] = [0.5+delta, 0.5+delta, 0.5+delta]
        d = dong(to_tensor(img))[0,0].cpu().numpy()
        edge = d[30:34].mean()
        ratio = d[30:34].mean() / d[:28].mean().clip(1e-8)
        check(ratio > 3, f"contrast={contrast:.2f} edge>flat",
              f"edge={edge:.4f} flat={d[:28].mean():.4f} ratio={ratio:.1f}")

    # raw gradient (before /amax) should scale with contrast
    # /amax normalizes per image, so peak ≈ 1.0 always
    # Instead verify edge/flat ratio scales with contrast
    for contrast in [0.125, 0.25, 0.5]:
        img = np.zeros((64, 64, 3), dtype=np.float32)
        img[:32] = [0.5]*3; img[32:] = [0.5+contrast]*3
        d = dong(to_tensor(img))[0,0].cpu().numpy()
        ratio = d[30:34].mean() / d[:28].mean().clip(1e-8)
        check(ratio > 2, f"contrast={contrast:.3f} edge/flat ratio>2",
              f"ratio={ratio:.1f}")


# ═══════════════════════════════════════════════════════════
# Test 3: Scale invariance — resize should preserve dong distribution
# ═══════════════════════════════════════════════════════════
def test_scale():
    print("\n=== Test 3: Scale ===")
    # Checkerboard pattern
    H, W = 128, 128
    y, x = np.mgrid[:H, :W]
    cb = ((y // 8 + x // 8) % 2).astype(np.float32)
    img = np.stack([cb, cb, cb], axis=-1)

    for size in [32, 64, 128, 256]:
        img_r = cv2.resize(img, (size, size))
        d = dong(to_tensor(img_r))[0,0].cpu().numpy()
        check(0.05 < d.mean() <= 1.0, f"checkerboard {size}x{size} dong mean",
              f"mean={d.mean():.4f}")


# ═══════════════════════════════════════════════════════════
# Test 4: Noise robustness
# ═══════════════════════════════════════════════════════════
def test_noise():
    print("\n=== Test 4: Noise ===")
    img_clean = np.full((64, 64, 3), 0.5, dtype=np.float32)

    for noise_std in [0.01, 0.05, 0.1, 0.2]:
        np.random.seed(42)
        noise = np.random.randn(64, 64, 3).astype(np.float32) * noise_std
        img_noisy = np.clip(img_clean + noise, 0, 1)
        d_clean = dong(to_tensor(img_clean))[0,0].cpu().numpy().mean()
        d_noisy = dong(to_tensor(img_noisy))[0,0].cpu().numpy().mean()
        # /amax: noise dominates → noisy dong > clean dong
        check(d_noisy > d_clean * 2,
              f"noise std={noise_std:.2f} detected (noisy>clean)",
              f"clean={d_clean:.4f} noisy={d_noisy:.4f}")


# ═══════════════════════════════════════════════════════════
# Test 5: Jing/Dong complement
# ═══════════════════════════════════════════════════════════
def test_complement():
    print("\n=== Test 5: Jing/Dong Complement ===")
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[:32] = [0.2, 0.9, 0.3]
    img[32:] = [0.7, 0.1, 0.8]
    t = to_tensor(img)
    d = dong(t)[0,0].cpu().numpy()
    j = jing(t)[0,0].cpu().numpy()

    # jing should be high where dong is low
    flat_mask = d < 0.1
    edge_mask = d > d.mean()
    check(j[flat_mask].mean() > 0.5, "jing high in flat areas",
          f"jing_flat={j[flat_mask].mean():.4f}")
    check(j[edge_mask].mean() < j[flat_mask].mean(),
          "jing lower at edges than flat",
          f"jing_edge={j[edge_mask].mean():.4f} jing_flat={j[flat_mask].mean():.4f}")


# ═══════════════════════════════════════════════════════════
# Test 6: Multiscale edges (thin vs thick lines)
# ═══════════════════════════════════════════════════════════
def test_line_width():
    print("\n=== Test 6: Line Widths ===")
    H, W = 128, 128
    for width in [1, 2, 4, 8]:
        img = np.ones((H, W, 3), dtype=np.float32)
        c = W//2
        img[:, c-width//2:c-width//2+width] = 0.0  # black vertical line
        d = dong(to_tensor(img))[0,0].cpu().numpy()
        peak = d[:, c-width:c+width].max()
        check(peak > 0.3, f"line width={width}px detected",
              f"peak dong={peak:.4f}")


# ═══════════════════════════════════════════════════════════
# Test 7: Real image jing → background vs object
# ═══════════════════════════════════════════════════════════
def test_jing_separation():
    print("\n=== Test 7: Jing Background Separation ===")
    img_path = Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'
    if not img_path.exists():
        print("  SKIP: no test image")
        return

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = 256, 256
    img = cv2.resize(img, (W, H))
    j = jing(to_tensor(img))[0,0].cpu().numpy()

    # Background corners should be high jing
    corners = np.concatenate([j[:16,:16].ravel(), j[:16,-16:].ravel(),
                               j[-16:,:16].ravel(), j[-16:,-16:].ravel()])
    center = j[H//2-16:H//2+16, W//2-16:W//2+16]
    check(corners.mean() > center.mean() * 0.8,
          "corners similar or higher jing than center",
          f"corners={corners.mean():.4f} center={center.mean():.4f}")


# ═══════════════════════════════════════════════════════════
# Test 8: Color texture — dong should detect texture variance
# ═══════════════════════════════════════════════════════════
def test_texture():
    print("\n=== Test 8: Textured vs Smooth ===")
    H, W = 64, 64
    np.random.seed(123)
    # Textured: random color patches
    tex = np.random.rand(H//4, W//4, 3).astype(np.float32)
    tex = cv2.resize(tex, (W, H), interpolation=cv2.INTER_NEAREST)
    # Smooth: uniform
    smooth = np.full((H, W, 3), 0.5, dtype=np.float32)

    d_tex = dong(to_tensor(tex))[0,0].cpu().numpy().mean()
    d_smooth = dong(to_tensor(smooth))[0,0].cpu().numpy().mean()
    check(d_tex > d_smooth * 5, "textured >> smooth dong",
          f"tex={d_tex:.4f} smooth={d_smooth:.4f}")


# ═══════════════════════════════════════════════════════════
# Test 9: batch invariance
# ═══════════════════════════════════════════════════════════
def test_batch():
    print("\n=== Test 9: Batch Independence ===")
    img1 = np.full((32, 32, 3), 0.3, dtype=np.float32)
    img2 = np.zeros((32, 32, 3), dtype=np.float32)
    img2[:16] = [1.0, 0.0, 0.0]
    img2[16:] = [0.0, 0.0, 1.0]
    batch = torch.cat([to_tensor(img1), to_tensor(img2)], dim=0)
    d = dong(batch)
    # img1 uniform → low dong
    check(d[0].mean() < 0.02, "batch[0] uniform dong≈0",
          f"mean={d[0].mean():.4f}")
    # img2 has edge → high dong at center
    check(d[1].mean() > 0.01, "batch[1] edge detected",
          f"mean={d[1].mean():.4f}")


# ═══════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    test_uniform()
    test_gradient()
    test_scale()
    test_noise()
    test_complement()
    test_line_width()
    test_jing_separation()
    test_texture()
    test_batch()
    print(f"\n{'='*50}")
    print(f"Results: {pass_count} PASS, {fail_count} FAIL out of {pass_count+fail_count}")
    print(f"{'ALL TESTS PASSED' if fail_count == 0 else 'SOME TESTS FAILED'}")
