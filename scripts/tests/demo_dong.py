"""Dong/Jing demo: side-by-side comparison on diverse patterns"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.operators import dong, jing

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent.parent / 'test_output'
OUT.mkdir(exist_ok=True)

def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEVICE)

def process(img_hwc):
    """Return dong, jing numpy arrays"""
    d = dong(to_tensor(img_hwc))[0,0].cpu().numpy()
    j = jing(to_tensor(img_hwc))[0,0].cpu().numpy()
    return d, j

# ═══════════════════════════════════════════
# Build test patterns
# ═══════════════════════════════════════════
S = 128
patterns = {}

# 1. Synthetic: red/blue sharp edge
img = np.zeros((S, S, 3), dtype=np.float32)
img[:, :S//2] = [1, 0, 0]
img[:, S//2:] = [0, 0, 1]
patterns['Sharp Edge'] = img

# 2. Synthetic: soft gradient
img = np.zeros((S, S, 3), dtype=np.float32)
ramp = np.linspace(0, 1, S).reshape(1, S, 1)
img[:] = ramp * [0, 1, 0]  # green ramp
patterns['Soft Gradient'] = img

# 3. Synthetic: checkerboard texture
y, x = np.mgrid[:S, :S]
cb = ((y // 8 + x // 8) % 2).astype(np.float32)
img = np.stack([cb * 0.8 + 0.1, cb * 0.3 + 0.2, cb * 0.9], axis=-1)
patterns['Texture'] = img

# 4. Synthetic: concentric rings
y, x = np.mgrid[:S, :S]
cx, cy = S//2, S//2
r = np.sqrt((x-cx)**2 + (y-cy)**2)
rings = (np.sin(r * 0.3) * 0.5 + 0.5).astype(np.float32)
img = np.stack([rings, 1-rings, rings*0.5], axis=-1)
patterns['Rings'] = img

# 5. Real photo
img_path = Path(__file__).resolve().parent.parent.parent / 'test_maccup.png'
if img_path.exists():
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (S, S))
    patterns['Real Photo'] = img

# ═══════════════════════════════════════════
# Render
# ═══════════════════════════════════════════
n = len(patterns)
fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))

for row, (name, img) in enumerate(patterns.items()):
    d, j = process(img)

    # Column 1: original
    axes[row, 0].imshow(np.clip(img, 0, 1))
    axes[row, 0].set_title(f'{name}\nOriginal')
    axes[row, 0].axis('off')

    # Column 2: dong — autoscale per image so weak signals visible
    vmax_d = max(d.max(), 0.05)
    axes[row, 1].imshow(d, cmap='hot', vmin=0, vmax=vmax_d)
    axes[row, 1].set_title(f'Dong [vmax={vmax_d:.2f}]\nmean={d.mean():.3f} max={d.max():.3f}')
    axes[row, 1].axis('off')

    # Column 3: jing — Blues reversed (high=dark? no, high=deep blue)
    axes[row, 2].imshow(j, cmap='Blues', vmin=0, vmax=1)
    axes[row, 2].set_title(f'Jing\nmean={j.mean():.3f} max={j.max():.3f}')
    axes[row, 2].axis('off')

plt.suptitle('Dong (change) vs Jing (still) — Physical Operators Demo',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out_path = OUT / 'demo_dong_jing.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out_path}')

# ═══════════════════════════════════════════
# Bonus: overlay on real photo
# ═══════════════════════════════════════════
if 'Real Photo' in patterns:
    img = patterns['Real Photo']
    d, j = process(img)
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_u8)
    axes[0].set_title('Original Photo')
    axes[0].axis('off')

    vmax_d = max(d.max(), 0.05)
    axes[1].imshow(d, cmap='hot', vmin=0, vmax=vmax_d)
    axes[1].set_title(f'Dong (change=gradient)')
    axes[1].axis('off')

    axes[2].imshow(j, cmap='Blues')
    axes[2].set_title('Jing (still=~flat)')
    axes[2].axis('off')

    # Overlay: dong=red, jing=blue — scale dong up for visibility
    d_scaled = np.clip(d / max(d.max(), 0.01), 0, 1)
    overlay = np.stack([d_scaled, np.zeros_like(d), j], axis=-1)
    axes[3].imshow(img_u8)
    axes[3].imshow(overlay, alpha=0.5)
    axes[3].set_title(f'Overlay (dong scaled)\nRed=Dong Blue=Jing')
    axes[3].axis('off')

    plt.tight_layout()
    out_path2 = OUT / 'demo_dong_real.png'
    fig.savefig(out_path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out_path2}')

print('\nDone. Open test_output/ to view images.')
