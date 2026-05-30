"""震多尺度测试"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; from src.operators import _zhen

imgs={}
# 窄边
e=np.ones((224,224,3),np.float32)*0.3; e[:,112]=0.8
imgs['1px边']=e
# 渐变边
gd=np.ones((224,224,3),np.float32)
for x in range(100,130): gd[:,x]=0.3+(x-100)/30*0.5
imgs['渐变边']=gd
# 杯子
cup=cv2.imread('test_maccup.png'); cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224)); imgs['杯子']=cup

fig,axes=plt.subplots(len(imgs),2,figsize=(8,2.5*len(imgs)))
for row,(title,img) in enumerate(imgs.items()):
    ct=torch.from_numpy(img[:,:,0].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    z=_zhen(ct)[0,0].cpu().numpy()
    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(title); axes[row,0].axis('off')
    p2,p98=np.percentile(z,2),np.percentile(z,98)
    axes[row,1].imshow(z,cmap='hot',vmin=p2,vmax=p98)
    axes[row,1].set_title(f'震 avg={z.mean():.2f} max={z.max():.2f}'); axes[row,1].axis('off')

plt.tight_layout()
out='test_output/test_zhen_v2.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'-> {out}')

# 杯子区域
ct=torch.from_numpy(cup[:,:,0].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
z=_zhen(ct)[0,0].cpu().numpy()
print(f'杯子: 杯边={z[55:70,70:150].mean():.3f}  杯内={z[80:140,70:140].mean():.3f}  bg={z[180:,20:60].mean():.3f}')
print(f'杯边>杯内? {"OK" if z[55:70,70:150].mean()>z[80:140,70:140].mean() else "FAIL"}')
