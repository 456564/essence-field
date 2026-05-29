"""艮多图测试: 纹理边界"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2, glob
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.operators import _gen

DATA='data/caltech101/101_ObjectCategories'
def ld(p):
    img=cv2.imread(p); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

pairs=[('杯子',ld('test_maccup.png'))]
for cat in ['butterfly','sunflower','chair','dollar_bill','faces','car_side',
            'BACKGROUND_Google','bonsai','watch','ketch']:
    fs=sorted(glob.glob(f'{DATA}/{cat}/*.jpg'))
    if fs: pairs.append((cat,ld(fs[len(fs)//2])))

fig,axes=plt.subplots(len(pairs),3,figsize=(12,2.2*len(pairs)))

for row,(name,img) in enumerate(pairs):
    ch=img[:,:,0]
    ct=torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    gen=_gen(ct)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(name,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(gen,2),np.percentile(gen,98)
    axes[row,1].imshow(gen,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'艮 avg={gen.mean():.2f} max={gen.max():.1f}',fontsize=9)
    axes[row,1].axis('off')

    # 纹理强度
    from src.operators import _box_filter
    lm=_box_filter(ct,k=15); tx=_box_filter((ct-lm)**2,k=15)[0,0].cpu().numpy()
    axes[row,2].imshow(tx,cmap='magma')
    axes[row,2].set_title(f'纹理 avg={tx.mean():.4f}',fontsize=9); axes[row,2].axis('off')

plt.tight_layout()
out='test_output/test_gen_multi.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'→ {out}')

print()
for name,img in pairs:
    ct=torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    gen=_gen(ct)[0,0].cpu().numpy()
    print(f'{name:>18s}: avg={gen.mean():.3f} max={gen.max():.1f}')
