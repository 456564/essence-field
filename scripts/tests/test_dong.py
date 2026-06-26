"""Dong/Jing operator tests: synthetic + real image validation"""
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
print(f"device: {DEVICE}")


def test_synthetic():
    """Synthetic: red/blue halves with vertical edge"""
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:, :W//2] = [1.0, 0.0, 0.0]       # left = pure red
    img[:, W//2:] = [0.0, 0.0, 1.0]       # right = pure blue
    # middle = sharp edge, high dong expected

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        d = dong(x)
        j = jing(x)

    d_np = d[0, 0].cpu().numpy()
    j_np = j[0, 0].cpu().numpy()

    edge_val = d_np[:, W//2].mean()
    flat_l = d_np[:, :W//2-2].mean()
    flat_r = d_np[:, W//2+2:].mean()

    print(f"\n=== Synthetic Dong ===")
    print(f"edge dong:   {edge_val:.4f}")
    print(f"flat left:   {flat_l:.4f}")
    print(f"flat right:  {flat_r:.4f}")
    print(f"edge/flat ratio: {edge_val / max(flat_l, flat_r):.1f}x")

    assert edge_val > flat_l * 3, f"edge({edge_val:.4f}) should >> flat({flat_l:.4f})"
    assert flat_l < 0.1, f"flat dong too high: {flat_l:.4f}"
    print("PASS: synthetic")


def test_real():
    """Real photo: mac cup, check dong/jing distribution"""
    img_path = Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'
    if not img_path.exists():
        print(f"SKIP: {img_path} not found")
        return

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (256, 256))
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        d = dong(x)
        j = jing(x)

    d_np = d[0, 0].cpu().numpy()
    j_np = j[0, 0].cpu().numpy()
    img_np = (img * 255).astype(np.uint8)

    print(f"\n=== Real Photo Dong/Jing ===")
    print(f"dong: min={d_np.min():.4f} max={d_np.max():.4f} mean={d_np.mean():.4f} median={np.median(d_np):.4f}")
    print(f"jing: min={j_np.min():.4f} max={j_np.max():.4f} mean={j_np.mean():.4f} median={np.median(j_np):.4f}")

    out_dir = Path(__file__).resolve().parent.parent.parent / 'test_output'
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    im1 = axes[1].imshow(d_np, cmap='hot')
    axes[1].set_title(f'Dong (change)\nmean={d_np.mean():.3f}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(j_np, cmap='Blues')
    axes[2].set_title(f'Jing (still)\nmean={j_np.mean():.3f}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    overlay = np.stack([d_np, np.zeros_like(d_np), j_np], axis=-1)
    axes[3].imshow(img_np, alpha=0.4)
    axes[3].imshow(overlay, alpha=0.6)
    axes[3].set_title('Overlay (red=dong blue=jing)')
    axes[3].axis('off')

    plt.tight_layout()
    out_path = out_dir / 'test_dong.png'
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    print("PASS: real")


def test_pipeline():
    """Verify pipeline integration"""
    from src.pipeline import PhysicalPipeline

    pipe = PhysicalPipeline().to(DEVICE)
    pipe.eval()

    x = torch.randn(1, 3, 128, 128).to(DEVICE).clamp(0, 1)
    with torch.no_grad():
        field = pipe(x)

    print(f"\n=== Pipeline ===")
    print(f"input:  {x.shape}")
    print(f"output: {field.shape}")
    assert field.shape == (1, 64, 128, 128), f"expected [1,64,128,128], got {field.shape}"
    print(f"params: {sum(p.numel() for p in pipe.parameters())}")
    print("PASS: pipeline")


if __name__ == '__main__':
    test_synthetic()
    test_real()
    test_pipeline()
    print("\nALL TESTS PASSED")
