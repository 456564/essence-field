"""兑算子测试: 开口/缺损 = 局部不连续"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; H,W=224,224
from src.operators import _dui

imgs={}

# 1. 暗洞 → 兑(开口)
dh=np.ones((H,W,3),np.float32)*0.7
yy,xx=np.ogrid[:H,:W]; r=np.sqrt((yy-112)**2+(xx-112)**2)
dh[r<30]=[0.1,0.1,0.1]
imgs['1_暗洞(开口)']=dh

# 2. 亮斑 → 也兑(缺损)
bs=np.ones((H,W,3),np.float32)*0.2
bs[r<30]=[0.9,0.9,0.9]
imgs['2_亮斑(突起)']=bs

# 3. 裂缝(细线)
cr=np.ones((H,W,3),np.float32)*0.7
cr[100:124,110:114]=[0.1,0.1,0.1]  # 水平裂缝
imgs['3_裂缝(细线)']=cr

# 4. 全平滑 → 兑≈0
sm=np.ones((H,W,3),np.float32)*0.5
imgs['4_全平滑(无缺口)']=sm

# 5. 杯子
cup=cv2.imread('test_maccup.png')
cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224))
imgs['5_杯子']=cup

fig,axes=plt.subplots(len(imgs),3,figsize=(12,2.2*len(imgs)))

for row,(title,img) in enumerate(imgs.items()):
    ch=img[:,:,0]
    ct=torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    dui=_dui(ct)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1),cmap='gray'); axes[row,0].set_title(title,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(dui,2),np.percentile(dui,98)
    axes[row,1].imshow(dui,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'兑 avg={dui.mean():.2f} max={dui.max():.1f}',fontsize=9)
    axes[row,1].axis('off')

    axes[row,2].plot(dui[H//2,:],'m-',lw=0.8)
    axes[row,2].set_title('中线剖面',fontsize=9)

plt.tight_layout()
out='test_output/test_dui_solo.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'→ {out}')

for title,img in imgs.items():
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    dui=_dui(ct)[0,0].cpu().numpy()
    cv=dui[H//2,W//2]
    print(f'{title:>20s}: center={cv:.1f} avg={dui.mean():.2f} max={dui.max():.1f}')
