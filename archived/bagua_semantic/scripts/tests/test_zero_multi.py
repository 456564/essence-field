"""零训练多图批量验证: 10张图t-SNE + 余弦分离度"""
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
for proj in pipe.operator_layer.projections.values():
    proj.weight.data.fill_(1.0)

def L(p):
    img=cv2.imread(str(p)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

DATA='data/caltech101/101_ObjectCategories'
pairs=[('Cup',L('test_maccup.png'))]
for cat in ['crab','butterfly','sunflower','car_side','faces','chair',
            'dollar_bill','watch','ketch','bonsai','BACKGROUND_Google','camera']:
    fs=sorted(glob.glob(f'{DATA}/{cat}/*.jpg'))
    if fs: pairs.append((cat,L(fs[len(fs)//2])))

# 计算每张图的分离度
results=[]
for name,img in pairs:
    x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad(): field=pipe(x); norms=field[0].norm(dim=0).cpu().numpy()

    # 物体/背景: norms > median
    mask=norms>np.median(norms)
    if mask.sum()<100 or (~mask).sum()<100:
        results.append((name,0,0,0,None)); continue

    f_obj=field[0,:,mask].reshape(64,-1).t().cpu().numpy()
    f_bg=field[0,:,~mask].reshape(64,-1).t().cpu().numpy()

    # 余弦相似度 (降采样)
    n=min(200,len(f_obj),len(f_bg))
    idx_o=np.random.choice(len(f_obj),n,replace=False)
    idx_b=np.random.choice(len(f_bg),n,replace=False)
    o=torch.tensor(f_obj[idx_o]); b=torch.tensor(f_bg[idx_b])
    o=F.normalize(o,dim=1); b=F.normalize(b,dim=1)
    intra=((o@o.t()).sum()-len(o))/(len(o)*(len(o)-1))
    cross=(o@b.t()).mean()

    # t-SNE
    from sklearn.manifold import TSNE
    combined=np.concatenate([o.numpy(),b.numpy()],axis=0)
    labels=np.array([0]*n+[1]*n)
    tsne=TSNE(n_components=2,perplexity=min(30,n-1),random_state=42).fit_transform(combined)
    results.append((name,intra.item(),cross.item(),intra.item()-cross.item(),(tsne,labels)))

# 画图
n=len(results)
cols=5; rows=(n+cols-1)//cols
fig=plt.figure(figsize=(cols*4,rows*3))

for idx,(name,intra,cross,diff,tsne_data) in enumerate(results):
    ax=fig.add_subplot(rows,cols,idx+1)
    if tsne_data is None:
        ax.text(0.5,0.5,f'{name}\ninsufficient data',ha='center',va='center')
        ax.axis('off'); continue
    tsne,labels=tsne_data
    ax.scatter(tsne[labels==0,0],tsne[labels==0,1],c='red',s=2,alpha=0.6)
    ax.scatter(tsne[labels==1,0],tsne[labels==1,1],c='blue',s=2,alpha=0.6)
    ax.set_title(f'{name}\nobj={intra:.2f} bg={cross:.2f} diff={diff:.3f}',fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out='test_output/zero_multi.png'
plt.savefig(out,dpi=100,bbox_inches='tight')
plt.close()
print(f'-> {out}')

# 排序
print(f'\n{"name":>18s}  {"intra":>6s}  {"cross":>6s}  {"diff":>7s}  {"sep?":>5s}')
print('-'*48)
for name,intra,cross,diff,_ in sorted(results,key=lambda x:-x[3]):
    ok='OK' if diff>0.05 else ('~' if diff>0.02 else 'FAIL')
    print(f'{name:>18s}  {intra:6.3f}  {cross:6.3f}  {diff:7.3f}  {ok:>5s}')
