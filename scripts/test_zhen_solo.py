"""震算子测试: 变化/激活度 = abs(laplacian)"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; H,W=224,224
from src.operators import _zhen

imgs={}

# 1. 锐利边缘
e=np.ones((H,W,3),np.float32)*0.3
e[:,112:]=0.8
imgs['1_锐边(高震)']=e

# 2. 平滑渐变
gd=np.ones((H,W,3),np.float32)
for x in range(W): gd[:,x]=x/W
imgs['2_渐变(低震)']=gd

# 3. 全平滑
sm=np.ones((H,W,3),np.float32)*0.5
imgs['3_全平滑(零)']=sm

# 4. 噪声
rn=np.ones((H,W,3),np.float32)*0.5
rn+=np.random.RandomState(42).randn(H,W,3).astype(np.float32)*0.3
rn=rn.clip(0,1)
imgs['4_噪声(高震)']=rn

# 5. 杯子
cup=cv2.imread('test_maccup.png')
cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224))
imgs['5_杯子']=cup

fig,axes=plt.subplots(len(imgs),3,figsize=(12,2.2*len(imgs)))

for row,(title,img) in enumerate(imgs.items()):
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    z=_zhen(ct)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1),cmap='gray'); axes[row,0].set_title(title,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(z,2),np.percentile(z,98)
    axes[row,1].imshow(z,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'震 avg={z.mean():.2f} max={z.max():.1f}',fontsize=9)
    axes[row,1].axis('off')

    axes[row,2].plot(z[H//2,:],'r-',lw=0.8)
    axes[row,2].set_title('中线剖面',fontsize=9)

plt.tight_layout()
out='test_output/test_zhen_solo.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'→ {out}')

for title,img in imgs.items():
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    z=_zhen(ct)[0,0].cpu().numpy()
    print(f'{title:>20s}: avg={z.mean():.3f} max={z.max():.1f}')
