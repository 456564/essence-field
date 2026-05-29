"""坎 — 真凹陷(辐射渐变暗圆) vs 均匀暗圆 vs 凸起"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device = 'cuda'
from src.operators import _kan

H, W = 224, 224
yy, xx = np.ogrid[:H, :W]
r = np.sqrt((yy-112)**2+(xx-112)**2)

imgs = {}

# 1. 辐射渐变凹陷: 中心最暗(0.05)→边缘亮(0.9), 过渡区50px
bowl = np.ones((H,W,3), np.float32)*0.9  # 亮背景
t = np.clip(r/50.0, 0, 1)               # 0=中心, 1=边缘
bowl[:,:,0] = 0.05 + 0.85*t; bowl[:,:,1] = bowl[:,:,0]; bowl[:,:,2] = bowl[:,:,0]                 # 中心0.05→边缘0.9
imgs['1_辐射凹陷\n(中心暗→外亮)'] = bowl

# 2. 均匀暗圆（对照）
flat_dark = np.ones((H,W,3), np.float32)*0.9
flat_dark[r<50] = 0.2
imgs['2_均匀暗圆\n(无凹陷)'] = flat_dark

# 3. 中心亮→外暗（凸起）
bump = np.ones((H,W,3), np.float32)*0.2  # 暗背景
t = np.clip(r/50.0, 0, 1)
bump[:,:,0] = 0.2 + 0.7*(1-t); bump[:,:,1] = bump[:,:,0]; bump[:,:,2] = bump[:,:,0]             # 中心0.9→边缘0.2
imgs['3_辐射凸起\n(中心亮→外暗)'] = bump

# 4. 杯子 — 看杯底阴影
cup = cv2.imread('test_maccup.png')
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup = cv2.resize(cup, (224,224))
imgs['4_杯子'] = cup

fig, axes = plt.subplots(len(imgs), 3, figsize=(12, 2.5*len(imgs)))

for row, (title, img) in enumerate(imgs.items()):
    ch = img[:,:,0]
    ch_t = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    kan = _kan(ch_t)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1), cmap='gray')
    axes[row,0].set_title(title, fontsize=9); axes[row,0].axis('off')

    p2, p98 = np.percentile(kan, 2), np.percentile(kan, 98)
    axes[row,1].imshow(kan, cmap='hot', vmin=p2, vmax=p98)
    axes[row,1].set_title(f'坎 avg={kan.mean():.1f} max={kan.max():.1f}', fontsize=9)
    axes[row,1].axis('off')

    axes[row,2].plot(kan[H//2,:], 'b-', linewidth=0.8)
    axes[row,2].set_title(f'中线剖面\n中心={kan[H//2,W//2]:.1f}', fontsize=9)

plt.tight_layout()
out = 'test_output/test_kan_v2.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

for title, img in imgs.items():
    ch_t = torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    kan = _kan(ch_t)[0,0].cpu().numpy()
    print(f'{title:>20s}: 中心坎={kan[H//2,W//2]:.1f}  avg={kan.mean():.1f}')
