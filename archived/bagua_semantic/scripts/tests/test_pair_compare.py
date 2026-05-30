"""两图对比: 8卦配方 + 自连接物体"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, torch.nn.functional as F, numpy as np, cv2, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline
pipe=BaguaPipeline().to(device).eval()
pipe.fusion.W_up.data=torch.eye(8,device=device)
pipe.fusion.W_dn.data=torch.eye(8,device=device)
for p in pipe.operator_layer.projections.values():
    p.weight.data=torch.clamp(1.0+0.3*torch.randn(8,1,1,1,device=device),min=0.1)

def L(p):
    img=cv2.imread(str(p)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

def get_recipe(field):
    norms=field[0].norm(dim=0); mask=norms>norms.quantile(0.7)
    r=torch.zeros(8,device=device)
    for i in range(8): r[i]=field[0,i*8:(i+1)*8][:,mask].norm(dim=0).mean()
    return (r/(r.sum()+1e-6)).cpu(), mask.cpu().numpy()

DATA='data/caltech101/101_ObjectCategories'
# 选三组对比
pairs=[
    ('cup1', f'{DATA}/cup/image_0001.jpg', 'cup2', f'{DATA}/cup/image_0002.jpg'),
    ('chair1',f'{DATA}/chair/image_0001.jpg','chair2',f'{DATA}/chair/image_0002.jpg'),
    ('bfly1',f'{DATA}/butterfly/image_0001.jpg','bfly2',f'{DATA}/butterfly/image_0002.jpg'),
    ('cam1', f'{DATA}/camera/image_0001.jpg','cam2',f'{DATA}/camera/image_0002.jpg'),
]

fig,axes=plt.subplots(4,4,figsize=(14,11))
opn=['乾Q','坤K','震Z','巽X','坎KA','离L','艮G','兑D']
c=['#FF6B35','#2D6A4F','#219EBC','#E9C46A','#023E8A','#D90429','#FB8B24','#8ECAE6']

for row,(n1,p1,n2,p2) in enumerate(pairs):
    for idx,(name,path) in enumerate([(n1,p1),(n2,p2)]):
        img=L(path)
        x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad(): field=pipe(x)
        recipe,mask=get_recipe(field)
        pass

        col=idx*2
        axes[row,col].imshow(img.clip(0,1)); axes[row,col].set_title(name,fontsize=8); axes[row,col].axis('off')
        axes[row,col+1].bar(range(8),recipe.numpy(),color=c)
        axes[row,col+1].set_xticks(range(8)); axes[row,col+1].set_xticklabels(opn,fontsize=6)
        axes[row,col+1].set_ylim(0,0.5)
        top=np.argsort(-recipe.numpy())[:3]
        rcp=' '.join(f'{opn[t]}={recipe[t]*100:.0f}%' for t in top)
        axes[row,col+1].set_title(rcp,fontsize=7)

    # 相似度: 全8卦 + 去坤乾纯6卦
    img1=L(p1); x1=torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().to(device)
    img2=L(p2); x2=torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        r1,_=get_recipe(pipe(x1)); r2,_=get_recipe(pipe(x2))
    sim8=(r1@r2/(r1.norm()*r2.norm()+1e-6)).item()
    r1p=r1.clone(); r2p=r2.clone()
    r1p[0]=r1p[1]=0; r2p[0]=r2p[1]=0  # 去掉乾(index0)坤(index1)
    sim6=(r1p@r2p/(r1p.norm()*r2p.norm()+1e-6)).item()
    print(f'{n1} vs {n2}: cos8={sim8:.3f}  cos6(去乾坤)={sim6:.3f}')

plt.tight_layout()
out='test_output/pair_compare.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'-> {out}')
