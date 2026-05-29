"""新算子 → 全管线 → 64维场验证(无训练, A=单位阵)"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline
pipe=BaguaPipeline().to(device).eval()

# A=单位阵 → 每个卦的8维只和自身交互
pipe.fusion.W_up.data=torch.eye(8,device=device); pipe.fusion.W_dn.data=torch.eye(8,device=device)

op_cn={'qian':'乾','kun':'坤','zhen':'震','xun':'巽','kan':'坎','li':'离','gen':'艮','dui':'兑'}

# 测试图
imgs={}
# 圆vs方(乾测试)
H=W=224; yy,xx=np.ogrid[:H,:W]; r=np.sqrt((yy-112)**2+(xx-112)**2)
c=np.ones((H,W,3),np.float32)*0.3; c[r<50]=0.8; imgs['圆(乾应高)']=c
sq=np.ones((H,W,3),np.float32)*0.3; sq[62:162,62:162]=0.8; imgs['方(乾应低)']=sq
# 杯子
cup=cv2.imread('test_maccup.png'); cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
imgs['杯子']=cv2.resize(cup,(224,224))

fig,axes=plt.subplots(len(imgs),10,figsize=(22,2.5*len(imgs)))

for row,(title,img) in enumerate(imgs.items()):
    x=torch.from_numpy(img.copy().astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        field=pipe(x)  # [1,64,H,W]

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(title,fontsize=8)
    axes[row,0].axis('off')

    # 每个卦的8维范数
    for i,name in enumerate(op_cn):
        v=field[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy()
        p2,p98=np.percentile(v,2),np.percentile(v,98)
        axes[row,i+1].imshow(v,cmap='hot',vmin=p2,vmax=p98)
        axes[row,i+1].set_title(f'{op_cn[name]}\navg={v.mean():.2f}',fontsize=7)
        axes[row,i+1].axis('off')
    axes[row,9].axis('off')

plt.tight_layout()
out='test_output/pipeline_v2.png'
plt.savefig(out,dpi=100,bbox_inches='tight')
plt.close()
print(f'→ {out}')

# 关键对比: 圆的乾是否 > 方的乾
for title,img in imgs.items():
    x=torch.from_numpy(img.copy().astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        field=pipe(x)
    print(f'\n{title}:')
    for i,name in enumerate(op_cn):
        v=field[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy()
        print(f'  {op_cn[name]}: avg={v.mean():.3f}  center={v[H//2,W//2]:.3f}')
