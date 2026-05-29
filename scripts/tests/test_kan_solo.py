"""坎算子测试：暗色低洼 + 梯度曲率"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from src.operators import _kan

H, W = 224, 224
imgs = {}
yy, xx = np.ogrid[:H, :W]
r = np.sqrt((yy-112)**2+(xx-112)**2)

# 1. 暗中心亮外围 — 低洼
dc = np.ones((H,W,3),np.float32)*0.8  # 亮背景
dc[r<50] = [0.2, 0.2, 0.3]           # 暗圆(水洼)
imgs['1_暗圆(水洼)'] = dc

# 2. 亮中心暗外围 — 突起
bc = np.ones((H,W,3),np.float32)*0.2  # 暗背景
bc[r<50] = [0.8, 0.8, 0.8]           # 亮圆
imgs['2_亮圆(突起)'] = bc

# 3. 条纹纹理 — 水流纹
stripe = np.ones((H,W,3),np.float32)*0.5
for i in range(0, W, 10):
    stripe[:, i:i+3] = [0.7, 0.7, 0.7] if i%20==0 else [0.3, 0.3, 0.4]
imgs['3_条纹(波)'] = stripe

# 4. 曲率圆
imgs['4_实心圆'] = dc  # same as 1

# 5. 杯子
cup = cv2.imread('test_maccup.png')
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup = cv2.resize(cup, (224,224))
imgs['5_杯子'] = cup

fig, axes = plt.subplots(len(imgs), 3, figsize=(12, 2.5*len(imgs)))

for row, (title, img) in enumerate(imgs.items()):
    ch = img[:,:,0]
    ch_t = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    kan = _kan(ch_t)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(title,fontsize=9)
    axes[row,0].axis('off')

    p2, p98 = np.percentile(kan, 2), np.percentile(kan, 98)
    axes[row,1].imshow(kan, cmap='hot', vmin=p2, vmax=p98)
    axes[row,1].set_title(f'坎 avg={kan.mean():.1f} max={kan.max():.1f}',fontsize=9)
    axes[row,1].axis('off')

    axes[row,2].plot(kan[H//2,:],'b-',linewidth=0.8)
    axes[row,2].set_title('中线剖面',fontsize=9)

plt.tight_layout()
out='test_output/test_kan_solo.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

for title, img in imgs.items():
    ch_t = torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    kan = _kan(ch_t)[0,0].cpu().numpy()
    print(f'{title:>20s}: center={kan[H//2,W//2]:.1f}  max={kan.max():.1f}  avg={kan.mean():.1f}')
