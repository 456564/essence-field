"""Real photo dong/jing test with detailed statistics"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.operators import dong, jing

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = __import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

img_path = __import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'
img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
H, W = img.shape[:2]
x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    d = dong(x)[0, 0].cpu().numpy()
    j = jing(x)[0, 0].cpu().numpy()

# ── Statistics ──
print(f"Image: {H}x{W}")
print(f"dong: min={d.min():.4f} max={d.max():.4f} mean={d.mean():.4f} median={np.median(d):.4f}")
print(f"jing: min={j.min():.4f} max={j.max():.4f} mean={j.mean():.4f} median={np.median(j):.4f}")

# Percentile breakdown
for p in [50, 75, 90, 95, 99]:
    print(f"  dong P{p}: {np.percentile(d, p):.4f}")

# Top-1% pixels
top1 = np.percentile(d, 99)
edge_pixels = (d > top1).sum()
print(f"\nPixels above P99 ({top1:.4f}): {edge_pixels} / {H*W} ({100*edge_pixels/(H*W):.1f}%)")

# ── Visualize ──
img_u8 = (img * 255).astype(np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: original + dong + jing
axes[0, 0].imshow(img_u8)
axes[0, 0].set_title(f'Original ({H}x{W})')
axes[0, 0].axis('off')

vmax_d = max(d.max(), 0.05)
im1 = axes[0, 1].imshow(d, cmap='hot', vmin=0, vmax=vmax_d)
axes[0, 1].set_title(f'Dong [vmax={vmax_d:.3f}]\nmean={d.mean():.3f} max={d.max():.3f}')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='dong')

im2 = axes[0, 2].imshow(j, cmap='Blues', vmin=0, vmax=1)
axes[0, 2].set_title(f'Jing\nmean={j.mean():.3f} max={j.max():.3f}')
axes[0, 2].axis('off')
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='jing')

# Row 2: overlay + dong histogram + dong thresholded
# Overlay
d_scaled = np.clip(d / max(d.max(), 0.01), 0, 1)
overlay = np.stack([d_scaled, np.zeros_like(d), j], axis=-1)
axes[1, 0].imshow(img_u8)
axes[1, 0].imshow(overlay, alpha=0.5)
axes[1, 0].set_title('Overlay: Red=Dong Blue=Jing')
axes[1, 0].axis('off')

# Histogram
axes[1, 1].hist(d.ravel(), bins=100, color='red', alpha=0.7, label='dong')
axes[1, 1].axvline(d.mean(), color='darkred', linestyle='--', label=f'mean={d.mean():.3f}')
axes[1, 1].axvline(np.median(d), color='orange', linestyle='--', label=f'median={np.median(d):.3f}')
axes[1, 1].set_title('Dong histogram')
axes[1, 1].legend()
axes[1, 1].set_xlabel('dong value')

# Thresholded: only show dong > P90
thresh = np.percentile(d, 90)
edge_map = (d > thresh).astype(np.float32)
axes[1, 2].imshow(img_u8, alpha=0.6)
axes[1, 2].imshow(edge_map, cmap='hot', alpha=0.7)
axes[1, 2].set_title(f'Dong > P90 ({thresh:.3f})\n{edge_map.sum():.0f} edge pixels')
axes[1, 2].axis('off')

plt.suptitle('Real Photo: Dong (change) & Jing (still)', fontsize=14, fontweight='bold')
plt.tight_layout()
out_path = OUT / 'test_real_dong.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out_path}")
