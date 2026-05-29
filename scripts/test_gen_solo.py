"""艮算子测试: 纹理突变边界"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; H,W=224,224; rng=np.random.RandomState(42)
from src.operators import _gen

imgs={}

# 1. 左平滑/右噪声 → 交界=艮
lr=np.ones((H,W,3),np.float32)*0.5
lr[:,:112]=0.5  # 左半平滑
ns=rng.randn(H,112,3).astype(np.float32)*0.3
lr[:,112:]=np.clip(0.5+ns,0,1)  # 右半噪声
imgs['1_平滑|噪声\n(边界=艮)']=lr

# 2. 全平滑 → 孑艮
sm=np.ones((H,W,3),np.float32)*0.5
imgs['2_全平滑\n(无边界)']=sm

# 3. 条纹 | 棋盘 — 两种纹理交界
tb=np.ones((H,W,3),np.float32)*0.5
for y in range(0,H//2,4): tb[y:y+2,:112]=[0.3,0.3,0.3]  # 左上条纹
for y in range(0,H//2,12):
    for x in range(112,W,12):
        tb[y:y+12,x:x+12]=0.3 if ((y//12)+(x//12))%2==0 else 0.7  # 右上棋盘
imgs['3_条纹|棋盘\n(双纹理)']=tb

# 4. 杯子
cup=cv2.imread('test_maccup.png')
cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224))
imgs['4_杯子']=cup

fig,axes=plt.subplots(len(imgs),4,figsize=(14,2.2*len(imgs)))

for row,(title,img) in enumerate(imgs.items()):
    ch=img[:,:,0]
    ct=torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    gen=_gen(ct)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(title,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(gen,2),np.percentile(gen,98)
    axes[row,1].imshow(gen,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'艮 avg={gen.mean():.2f} max={gen.max():.1f}',fontsize=9)
    axes[row,1].axis('off')

    # 纹理强度(局部方差)
    import torch.nn.functional as F
    from src.operators import _box_filter
    lm=_box_filter(ct,k=7)
    tx=_box_filter((ct-lm)**2,k=7)[0,0].cpu().numpy()
    axes[row,2].imshow(tx,cmap='magma')
    axes[row,2].set_title(f'纹理强度 avg={tx.mean():.4f}',fontsize=9); axes[row,2].axis('off')

    axes[row,3].plot(gen[H//2,:],'b-',lw=0.8)
    axes[row,3].set_title('中线剖面',fontsize=9)

plt.tight_layout()
out='test_output/test_gen_solo.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'→ {out}')

for title,img in imgs.items():
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    gen=_gen(ct)[0,0].cpu().numpy()
    print(f'{title:>20s}: avg={gen.mean():.3f} max={gen.max():.1f}')
