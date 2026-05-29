"""双投影融合可视化"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, cv2, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; from src.pipeline import BaguaPipeline

opn=['qian','kun','zhen','xun','kan','li','gen','dui']
opc={'qian':'Q','kun':'K','zhen':'ZH','xun':'X','kan':'KA','li':'L','gen':'G','dui':'D'}

pipe=BaguaPipeline().to(device).eval()
ckpt=torch.load(sorted(glob.glob('checkpoints_fixedcolor/bootstrap_epoch*.pth'))[-1],map_location=device)
pipe.fusion.W_up.data=ckpt.get('W_up'); pipe.fusion.W_dn.data=ckpt.get('W_dn')
pipe.operator_layer.projections.load_state_dict(ckpt['proj'])

def L(p):
    img=cv2.imread(str(p)); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img,(224,224))

cup=L('test_maccup.png')
btr=L('data/caltech101/101_ObjectCategories/butterfly/image_0002.jpg')
crab=L('data/caltech101/101_ObjectCategories/crab/image_0001.jpg')

fig,axes=plt.subplots(4,4,figsize=(16,12))

# W_up, W_dn
wu=ckpt['W_up'].cpu().numpy()
im=axes[0,0].imshow(wu,cmap='RdYlGn',vmin=0,vmax=max(wu.max(),1))
axes[0,0].set_title('W_up'); plt.colorbar(im,ax=axes[0,0],shrink=0.7)
wd=ckpt['W_dn'].cpu().numpy()
im=axes[0,1].imshow(wd,cmap='RdYlGn',vmin=0,vmax=max(wd.max(),1))
axes[0,1].set_title('W_dn'); plt.colorbar(im,ax=axes[0,1],shrink=0.7)
axes[0,2].axis('off'); axes[0,3].axis('off')

c=['#FF6B35','#2D6A4F','#219EBC','#E9C46A','#023E8A','#D90429','#FB8B24','#8ECAE6']
cm=np.array([[1,0.4,0.2],[0.2,0.7,0.3],[0.1,0.6,0.9],[0.9,0.8,0.3],
             [0,0.3,0.7],[0.85,0.1,0.1],[1,0.6,0.2],[0.4,0.7,0.95]])

for idx,(title,img) in enumerate([('Cup',cup),('Butterfly',btr),('Crab',crab)]):
    row=idx+1
    x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad(): field=pipe(x); norms=field[0].norm(dim=0).cpu().numpy()

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(title); axes[row,0].axis('off')

    axes[row,1].imshow(norms,cmap='hot'); axes[row,1].set_title(f'||field||'); axes[row,1].axis('off')

    vals=[field[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy().mean() for i in range(8)]
    axes[row,2].bar(range(8),vals,color=c); axes[row,2].set_xticks(range(8))
    axes[row,2].set_xticklabels([opc[n] for n in opn],fontsize=7)
    axes[row,2].set_ylim(0,max(vals)*1.2)

    strength=np.stack([field[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy() for i in range(8)])
    am=np.argmax(strength,axis=0)
    rgb=np.zeros((224,224,3))
    for i in range(8): rgb[am==i]=cm[i]
    axes[row,3].imshow(rgb); axes[row,3].set_title('argmax trigram'); axes[row,3].axis('off')

plt.tight_layout()
out='test_output/fusion_viz.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'-> {out}')
print(f'W_up diag mean: {wu.diagonal().mean():.3f}')
print(f'W_dn diag mean: {wd.diagonal().mean():.3f}')
