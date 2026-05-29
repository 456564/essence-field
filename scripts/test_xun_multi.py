"""巽多图测试: 真实图中方向性纹理"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2, glob
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.operators import _xun

DATA='data/caltech101/101_ObjectCategories'
def ld(p):
    img=cv2.imread(p); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

pairs=[('杯子',ld('test_maccup.png'))]
for cat in ['butterfly','sunflower','chair','dollar_bill','faces','car_side',
            'BACKGROUND_Google','ketch','bonsai','watch']:
    fs=sorted(glob.glob(f'{DATA}/{cat}/*.jpg'))
    if fs: pairs.append((cat,ld(fs[len(fs)//2])))

fig,axes=plt.subplots(len(pairs),3,figsize=(12,2.2*len(pairs)))

for row,(name,img) in enumerate(pairs):
    ch=img[:,:,0]
    ct=torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    xu=_xun(ct)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(name,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(xu,2),np.percentile(xu,98)
    axes[row,1].imshow(xu,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'巽 avg={xu.mean():.2f} max={xu.max():.1f}',fontsize=9)
    axes[row,1].axis('off')

    import torch.nn.functional as F
    sk=torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype=torch.float32,device=device)
    gx=F.conv2d(ct,sk,padding=1); gy=F.conv2d(ct,sk.transpose(2,3),padding=1)
    en=torch.sqrt(gx**2+gy**2+1e-6)[0,0].cpu().numpy()
    axes[row,2].imshow(en,cmap='magma')
    axes[row,2].set_title(f'能量 avg={en.mean():.3f}',fontsize=9); axes[row,2].axis('off')

plt.tight_layout()
out='test_output/test_xun_multi.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'→ {out}')

print()
for name,img in pairs:
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    xu=_xun(ct)[0,0].cpu().numpy()
    print(f'{name:>18s}: avg={xu.mean():.3f} max={xu.max():.1f}')
