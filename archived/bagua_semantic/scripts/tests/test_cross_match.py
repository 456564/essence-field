"""跨图匹配: 同类物体64维质心是否相近"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, torch.nn.functional as F, numpy as np, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline

pipe=BaguaPipeline().to(device).eval()
pipe.fusion.W_up.data=torch.eye(8,device=device)
pipe.fusion.W_dn.data=torch.eye(8,device=device)
for p in pipe.operator_layer.projections.values():
    p.weight.data=torch.clamp(1.0+0.3*torch.randn(8,1,1,1,device=device),min=0.1)

def load(p):
    import cv2
    img=cv2.imread(str(p)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

DATA='data/caltech101/101_ObjectCategories'

# 选4类, 每类5张
categories={'cup':5,'chair':5,'butterfly':5,'camera':5}
embeddings={}  # cat -> list of 64-dim centroids

for cat,n in categories.items():
    files=sorted(glob.glob(f'{DATA}/{cat}/*.jpg'))[:n]
    embs=[]
    for f in files:
        img=load(f)
        x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad(): field=pipe(x)
        # 物体像素: norms > median
        norms=field[0].norm(dim=0)
        thresh=norms.quantile(0.7)  # 顶30% = 纯物体像素
        mask=norms>thresh
        if mask.sum()<100: continue
        # 8卦配方比例: 每卦8维L2范数 → 8维向量 → 归一化 = 百分比配方
        recipe=torch.zeros(8,device=device)
        for i in range(8): recipe[i]=field[0,i*8:(i+1)*8][:,mask].norm(dim=0).mean()
        recipe=recipe/(recipe.sum()+1e-6)
        embs.append(recipe.cpu())  # [8]
    if embs: embeddings[cat]=embs

# 计算相似度矩阵
all_cats=list(embeddings.keys())
n_total=sum(len(v) for v in embeddings.values())
sim=np.zeros((n_total,n_total))
labels=[]
for cat in all_cats:
    labels.extend([cat]*len(embeddings[cat]))

idx=0
for ci,cat_i in enumerate(all_cats):
    for ei in range(len(embeddings[cat_i])):
        jdx=0
        for cj,cat_j in enumerate(all_cats):
            for ej in range(len(embeddings[cat_j])):
                a=F.normalize(embeddings[cat_i][ei],dim=0)
                b=F.normalize(embeddings[cat_j][ej],dim=0)
                sim[idx,jdx]=(a*b).sum().item()
                jdx+=1
        idx+=1

# 可视化
fig,ax=plt.subplots(figsize=(8,7))
im=ax.imshow(sim,cmap='RdYlGn',vmin=0.5,vmax=1)
# 标注类别边界
x=0
ticks=[]
for cat in all_cats:
    n_c=len(embeddings[cat])
    ax.axhline(x-0.5,color='white',lw=2)
    ax.axvline(x-0.5,color='white',lw=2)
    ticks.append((x+n_c//2,cat))
    x+=n_c
ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks],fontsize=9)
ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks],fontsize=9)
ax.set_title('跨图: 8卦配方余弦相似度\n同类对角块应绿(高)',fontsize=11)
plt.colorbar(im,ax=ax,shrink=0.8)

plt.tight_layout()
out='test_output/cross_match.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'-> {out}')

# 统计
print(f'\n{"类别":>12s}  {"类内均值":>8s}  {"类间均值":>8s}  {"差距":>6s}')
for cat in all_cats:
    n_c=len(embeddings[cat])
    start=sum(len(embeddings[c]) for c in all_cats if c!=cat)
    intra_vals=[]; cross_vals=[]
    for i in range(n_c):
        for j in range(i+1,n_c):
            intra_vals.append(sim[sum(len(embeddings[c]) for c in all_cats[:all_cats.index(cat)])+i,
                                    sum(len(embeddings[c]) for c in all_cats[:all_cats.index(cat)])+j])
    for i in range(n_c):
        for j in range(sum(len(embeddings[c]) for c in all_cats[:all_cats.index(cat)])):
            cross_vals.append(sim[sum(len(embeddings[c]) for c in all_cats[:all_cats.index(cat)])+i,j])
        for j in range(sum(len(embeddings[c]) for c in all_cats[:all_cats.index(cat)])+n_c,n_total):
            cross_vals.append(sim[sum(len(embeddings[c]) for c in all_cats[:all_cats.index(cat)])+i,j])
    print(f'{cat:>12s}  {np.mean(intra_vals):8.3f}  {np.mean(cross_vals):8.3f}  {np.mean(intra_vals)-np.mean(cross_vals):+.3f}')
