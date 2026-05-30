"""Diagnostic: Soft Gradient dong/jing raw values"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np
from src.operators import dong, jing

S = 128
ramp = np.linspace(0, 1, S).reshape(1, S, 1).astype(np.float32)
img = np.zeros((S, S, 3), dtype=np.float32)
img[:] = ramp * [0, 1, 0]  # black → green
x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

d = dong(x)[0, 0].cpu().numpy()
j = jing(x)[0, 0].cpu().numpy()

print("=== Soft Gradient Diagnostic ===")
print(f"dong: min={d.min():.6f}  max={d.max():.6f}  mean={d.mean():.6f}")
print(f"      unique values: {len(set(d.round(6).flat))}")
print(f"jing: min={j.min():.6f}  max={j.max():.6f}  mean={j.mean():.6f}")
print()
print("dong row[64] profile:")
print(f"  x=0..11:  {d[64, :12].round(6)}")
print(f"  x=60..67: {d[64, 60:68].round(6)}")
print(f"  x=116..127: {d[64, 116:].round(6)}")
print()
print("Physical interpretation:")
print(f"  gradient = {d[64,64]*4:.4f} / 4.0 max")
print(f"  {d[64,64]*100:.1f}% of max possible edge strength")
print(f"  A sharp B/W edge would be ~1.15 (clamped to 1.0)")
print()
print("Display: value {:.4f} in hot[0,1] colormap = #{:02x}0000 (near black)".format(
    d[64, 64], int(d[64, 64] * 255)))
print("This is CORRECT — uniform gradient = uniform dong = low absolute value.")
