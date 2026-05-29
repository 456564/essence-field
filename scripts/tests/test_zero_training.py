"""零训练验证: W=I, 投影=1, 看64维场天然聚类"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, torch.nn.functional as F, numpy as np, cv2, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline
opn=['qian','kun','zhen','xun','kan','li','gen','dui']
opc={'qian':'Q','kun':'K','zhen':'ZH','xun':'X','kan':'KA','li':'L','gen':'G','dui':'D'}

pipe=BaguaPipeline().to(device).eval()

# 零训练: W_up=W_dn=I, 投影权重全1
pipe.fusion.W_up.data=torch.eye(8,device=device)
pipe.fusion.W_dn.data=torch.eye(8,device=device)
for proj in pipe.operator_layer.projections.values():
    proj.weight.data.fill_(1.0)  # 每通道等权直通

def load(p):
    img=cv2.imread(str(p)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

cup=load('test_maccup.png')
btr=load('data/caltech101/101_ObjectCategories/butterfly/image_0002.jpg')
crab=load('data/caltech101/101_ObjectCategories/crab/image_0001.jpg')

fig,axes=plt.subplots(3,4,figsize=(16,10))
c=['#FF6B35','#2D6A4F','#219EBC','#E9C46A','#023E8A','#D90429','#FB8B24','#8ECAE6']
cm=np.array([[1,0.4,0.2],[0.2,0.7,0.3],[0.1,0.6,0.9],[0.9,0.8,0.3],
             [0,0.3,0.7],[0.85,0.1,0.1],[1,0.6,0.2],[0.4,0.7,0.95]])

for idx,(title,img) in enumerate([('Cup',cup),('Butterfly',btr),('Crab',crab)]):
    x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad(): field=pipe(x); norms=field[0].norm(dim=0).cpu().numpy()

    axes[idx,0].imshow(img.clip(0,1)); axes[idx,0].set_title(title); axes[idx,0].axis('off')
    axes[idx,1].imshow(norms,cmap='hot',vmin=norms.min(),vmax=norms.max())
    axes[idx,1].set_title(f'||field|| (zero-train)\nmask={norms.mean():.0f}'); axes[idx,1].axis('off')

    vals=[field[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy().mean() for i in range(8)]
    axes[idx,2].bar(range(8),vals,color=c); axes[idx,2].set_xticks(range(8))
    axes[idx,2].set_xticklabels([opc[n] for n in opn],fontsize=7)

    # 像素聚类: 取2个区域的64维向量
    H=W=224
    if title=='Cup':
        r1=(80,140,70,140)  # cup interior
        r2=(180,210,20,80)  # background
    elif title=='Butterfly':
        r1=(60,120,80,160)  # wing
        r2=(180,210,20,80)  # background
    else:
        r1=(40,120,60,160)  # crab body
        r2=(180,210,20,80)  # background

    f1=field[0,:,r1[0]:r1[1],r1[2]:r1[3]].reshape(64,-1).t().cpu().numpy()
    f2=field[0,:,r2[0]:r2[1],r2[2]:r2[3]].reshape(64,-1).t().cpu().numpy()

    # t-SNE降到2D看聚类
    from sklearn.manifold import TSNE
    combined=np.concatenate([f1[::4],f2[::4]],axis=0)  # 降采样
    labels=np.array([0]*len(f1[::4])+[1]*len(f2[::4]))
    tsne=TSNE(n_components=2,perplexity=30,random_state=42).fit_transform(combined)

    axes[idx,3].scatter(tsne[labels==0,0],tsne[labels==0,1],c='red',s=1,alpha=0.5,label='obj')
    axes[idx,3].scatter(tsne[labels==1,0],tsne[labels==1,1],c='blue',s=1,alpha=0.5,label='bg')
    axes[idx,3].set_title('t-SNE: obj(red) vs bg(blue)'); axes[idx,3].legend(fontsize=7)

    # 余弦相似度
    c1=F.normalize(torch.tensor(f1),dim=1)
    c2=F.normalize(torch.tensor(f2),dim=1)
    intra=((c1@c1.t()).sum()-len(c1))/(len(c1)*(len(c1)-1))
    cross=(c1@c2.t()).mean()
    print(f'{title}: obj内余弦={intra:.3f}  cross={cross:.3f}  diff={intra-cross:.3f}')

plt.tight_layout()
out='test_output/zero_training.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'-> {out}')
