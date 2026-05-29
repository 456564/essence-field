"""场自连接: 像素之间通过64维签名相似度自组物体"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, cv2, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline

pipe=BaguaPipeline().to(device).eval()
pipe.fusion.W_up.data=torch.eye(8,device=device)
pipe.fusion.W_dn.data=torch.eye(8,device=device)
for p in pipe.operator_layer.projections.values(): p.weight.data.fill_(1.0)

def L(p):
    img=cv2.imread(str(p)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

# 测试图
imgs=[('Cup',L('test_maccup.png')),
      ('Crab',L('data/caltech101/101_ObjectCategories/crab/image_0001.jpg')),
      ('Butterfly',L('data/caltech101/101_ObjectCategories/butterfly/image_0002.jpg')),
      ('Camera',L('data/caltech101/101_ObjectCategories/camera/image_0001.jpg')),
      ('Watch',L('data/caltech101/101_ObjectCategories/watch/image_0001.jpg')),
      ('Faces',L('data/caltech101/101_ObjectCategories/faces/image_0001.jpg'))]

def segment_by_field(field, cos_thresh=0.8, min_size=200):
    """场自连接: 邻域余弦相似度 + 连通域 = 物体分割"""
    H,W=field.shape[2],field.shape[3]
    f=field[0].reshape(64,-1).t()  # [H*W, 64]
    norms=f.norm(dim=1).cpu().numpy()
    median=np.median(norms)

    # 归一化
    fn=torch.nn.functional.normalize(f,dim=1).view(H,W,64)

    # 8邻域相似度: 用偏移算(快)
    offsets=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    link_count=np.zeros((H,W),dtype=np.int32)
    for dy,dx in offsets:
        rolled=torch.roll(fn,(dy,dx),dims=(0,1))
        sim=(fn*rolled).sum(dim=2).cpu().numpy()  # 余弦相似度
        # 边界不计数
        if dy<0: sim[:abs(dy)]=0
        if dy>0: sim[-dy:]=0
        if dx<0: sim[:,:abs(dx)]=0
        if dx>0: sim[:,-dx:]=0
        link=(sim>cos_thresh) & (norms.reshape(H,W)>median) & (sim>0)
        link_count+=link.astype(np.int32)

    # 连通域: BFS
    visited=np.zeros((H,W),dtype=bool)
    labels=np.full((H,W),-1,dtype=np.int32)
    label=0
    components=[]

    for y in range(H):
        for x in range(W):
            if visited[y,x] or norms[y*W+x]<=median: continue
            # BFS
            queue=[(y,x)]; visited[y,x]=True; pixels=[]
            while queue:
                cy,cx=queue.pop(0); pixels.append((cy,cx))
                for dy,dx in offsets:
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<H and 0<=nx<W and not visited[ny,nx]:
                        if link_count[ny,nx]>=2:  # 至少2个邻居同意
                            visited[ny,nx]=True; queue.append((ny,nx))
            if len(pixels)>=min_size:
                for cy,cx in pixels: labels[cy,cx]=label
                components.append((label, len(pixels)))
                label+=1

    return norms, median, labels, link_count, components

fig,axes=plt.subplots(len(imgs),4,figsize=(14,3.5*len(imgs)))

for row,(name,img) in enumerate(imgs):
    x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad(): field=pipe(x)
    norms,median,labels,links,comps=segment_by_field(field)

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(name,fontsize=10)
    axes[row,0].axis('off')

    axes[row,1].imshow(norms.reshape(224,224),cmap='hot',vmin=norms.min(),vmax=norms.max())
    axes[row,1].set_title(f'||field|| median={median:.0f}',fontsize=9); axes[row,1].axis('off')

    # 连通域着色
    colors=plt.cm.tab20(np.linspace(0,1,20))
    rgb=np.ones((224,224,3))*0.2  # 背景灰
    for lid,sz in sorted(comps,key=lambda x:-x[1])[:10]:
        c=colors[lid%20]
        rgb[labels==lid]=c[:3]
    axes[row,2].imshow(rgb); axes[row,2].set_title(f'{len(comps)} objects\n{len(comps)} components',fontsize=9)
    axes[row,2].axis('off')

    # 叠加原图
    over=img.copy()*0.4
    for lid,sz in sorted(comps,key=lambda x:-x[1])[:10]:
        c=colors[lid%20]
        mask=(labels==lid)
        over[mask]=over[mask]*0.3+np.array(c[:3])*0.7
    axes[row,3].imshow(over.clip(0,1)); axes[row,3].set_title('overlay',fontsize=9)
    axes[row,3].axis('off')

    # 打印配方 (64维→8卦强度)
    if comps:
        big=comps[0]
        opn=['qian','kun','zhen','xun','kan','li','gen','dui']
        opc={'qian':'乾','kun':'坤','zhen':'震','xun':'巽','kan':'坎','li':'离','gen':'艮','dui':'兑'}
        gua_strength=[field[0,i*8:(i+1)*8,labels==big[0]].norm(dim=0).mean().item() for i in range(8)]
        top=np.argsort(-np.array(gua_strength))[:4]
        gm=gua_strength
        recipe='+'.join(f'{opc[opn[t]]}{gm[t]/gm[top[0]]*100:.0f}%' for t in top)
        print(f'{name}: {len(comps)} obj, 最大={big[1]}px, 配方={recipe}')

plt.tight_layout()
out='test_output/self_link.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'\n-> {out}')
