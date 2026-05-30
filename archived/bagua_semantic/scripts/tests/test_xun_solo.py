"""巽算子测试：方向集中度 × 梯度能量"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; H,W=224,224; rng=np.random.RandomState(42)
imgs={}

# 1. 水平线
hl=np.ones((H,W,3),np.float32)*0.5
for y in range(0,H,6): hl[y:y+2,:]=[0.3,0.3,0.3]
imgs['1_水平线(高)']=hl

# 2. 随机噪声
rn=np.ones((H,W,3),np.float32)*0.5
rn+=rng.randn(H,W,3).astype(np.float32)*0.3; rn=rn.clip(0,1)
imgs['2_噪声(低)']=rn

# 3. 平滑渐变
gd=np.ones((H,W,3),np.float32)
for x in range(W): gd[:,x]=[x/W]*3
imgs['3_渐变(零)']=gd

# 4. 棋盘
ck=np.zeros((H,W,3),np.float32)
for y in range(0,H,16):
    for x in range(0,W,16):
        ck[y:y+16,x:x+16]=0.3 if ((y//16)+(x//16))%2==0 else 0.7
imgs['4_棋盘(中)']=ck

# 5. 杯子
cup=cv2.imread('test_maccup.png')
cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224))
imgs['5_杯子']=cup

from src.operators import _xun

fig,axes=plt.subplots(len(imgs),4,figsize=(14,2.2*len(imgs)))

for row,(title,img) in enumerate(imgs.items()):
    ch=img[:,:,0]
    ct=torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    xu=_xun(ct)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(title,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(xu,2),np.percentile(xu,98)
    axes[row,1].imshow(xu,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'巽 avg={xu.mean():.2f} max={xu.max():.1f}',fontsize=9)
    axes[row,1].axis('off')

    # 能量(梯度幅值)
    import torch.nn.functional as F
    sk=torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype=torch.float32,device=device)
    gx=F.conv2d(ct,sk,padding=1); gy=F.conv2d(ct,sk.transpose(2,3),padding=1)
    energy=torch.sqrt(gx**2+gy**2+1e-6)[0,0].cpu().numpy()
    axes[row,2].imshow(energy,cmap='magma')
    axes[row,2].set_title(f'能量(梯度) avg={energy.mean():.3f}',fontsize=9)
    axes[row,2].axis('off')

    axes[row,3].plot(xu[H//2,:],'g-',lw=0.8)
    axes[row,3].set_title('中线剖面',fontsize=9)

plt.tight_layout()
out='test_output/test_xun_solo.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'→ {out}')

for title,img in imgs.items():
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    xu=_xun(ct)[0,0].cpu().numpy()
    sk=torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype=torch.float32,device=device)
    gx=F.conv2d(ct,sk,padding=1); gy=F.conv2d(ct,sk.transpose(2,3),padding=1)
    en=torch.sqrt(gx**2+gy**2+1e-6)[0,0].cpu().numpy()
    print(f'{title:>15s}: 巽 avg={xu.mean():.3f} max={xu.max():.1f} | 能量 avg={en.mean():.3f}')
