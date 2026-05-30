"""乾算子测试: 完整/圆满度"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; H,W=224,224
from src.operators import _qian
imgs={}

# 1. 实心圆
yy,xx=np.ogrid[:H,:W]; r=np.sqrt((yy-112)**2+(xx-112)**2)
c=np.ones((H,W,3),np.float32)*0.3; c[r<50]=0.8
imgs['1_圆(完整)']=c

# 2. 方块
sq=np.ones((H,W,3),np.float32)*0.3; sq[62:162,62:162]=0.8
imgs['2_方块(不圆)']=sq

# 3. 星形(不完整)
st=np.ones((H,W,3),np.float32)*0.3
th=np.arctan2(yy-112,xx-112); pts=5; R=40+15*np.cos(pts*th)
st[np.sqrt((yy-112)**2+(xx-112)**2)<R]=0.8
imgs['3_星形(不整)']=st

# 4. 全平滑(无圆)
imgs['4_全平滑']=np.ones((H,W,3),np.float32)*0.5

# 5. 杯子
cup=cv2.imread('test_maccup.png')
cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224))
imgs['5_杯子']=cup

fig,axes=plt.subplots(len(imgs),3,figsize=(12,2.2*len(imgs)))

for row,(title,img) in enumerate(imgs.items()):
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    qian=_qian(ct)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1),cmap='gray'); axes[row,0].set_title(title,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(qian,2),np.percentile(qian,98)
    axes[row,1].imshow(qian,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'乾 avg={qian.mean():.1f} max={qian.max():.0f}',fontsize=9)
    axes[row,1].axis('off')

    axes[row,2].plot(qian[H//2,:],'r-',lw=0.8)
    axes[row,2].set_title('中线剖面',fontsize=9)

plt.tight_layout()
out='test_output/test_qian_solo.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'→ {out}')

for title,img in imgs.items():
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    qian=_qian(ct)[0,0].cpu().numpy()
    print(f'{title:>20s}: avg={qian.mean():.1f} max={qian.max():.0f}')
