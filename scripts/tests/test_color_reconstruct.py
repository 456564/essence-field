"""色块还原测试: RGB纯色→8算子→场→融合色→还原原色? """
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline
from src.visualize import blended_gua_response

pipe=BaguaPipeline().to(device).eval()
pipe.fusion.W_up.data=torch.eye(8,device=device)
pipe.fusion.W_dn.data=torch.eye(8,device=device)
for p in pipe.operator_layer.projections.values(): p.weight.data=torch.clamp(1.0+0.3*torch.randn(8,1,1,1,device=device),min=0.1)

# 8纯色块 + 白 + 黑
colors=[(1,0,0,'赤'),(0,1,0,'绿'),(0,0,1,'蓝'),(1,1,1,'白'),
        (1,1,0,'黄'),(0,1,1,'青'),(1,0,1,'紫'),(0,0,0,'黑'),
        (0.5,0.2,0,'棕'),(0.2,0.5,0.8,'灰蓝')]

H,W=64,64
imgs=[]
for r,g,b,name in colors:
    img=np.ones((H,W,3),np.float32)*[r,g,b]
    imgs.append((name,img))

fig,axes=plt.subplots(len(imgs),4,figsize=(12,2*len(imgs)))
for row,(name,img) in enumerate(imgs):
    x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        field=pipe(x)
        raw=pipe.operator_layer.base_ops(x)

    # 8卦强度
    strengths=[field[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy().mean() for i in range(8)]
    opn=['乾坤震巽坎离艮兑']
    top=np.argsort(-np.array(strengths))[:3]
    recipe=' '.join(f'{opn[0][t]}={strengths[t]:.1f}' for t in top)

    axes[row,0].imshow(img); axes[row,0].set_title(f'{name}原色',fontsize=8); axes[row,0].axis('off')
    blended=blended_gua_response(field)
    axes[row,1].imshow(blended); axes[row,1].set_title(f'融合色\n{recipe}',fontsize=8); axes[row,1].axis('off')

    # 各卦强度条
    axes[row,2].bar(range(8),strengths,color=['#FF6B35','#2D6A4F','#219EBC','#E9C46A','#023E8A','#D90429','#FB8B24','#8ECAE6'])
    axes[row,2].set_xticks(range(8)); axes[row,2].set_xticklabels(list('乾坤震巽坎离艮兑'),fontsize=6)

    # 比较: 原始RGB vs 融合色的RGB
    orig_rgb=img[0,0]
    blen_rgb=blended[32,32].astype(np.float32)/255
    axes[row,3].axis('off')
    axes[row,3].text(0.1,0.8,f'原始RGB=({orig_rgb[0]:.2f},{orig_rgb[1]:.2f},{orig_rgb[2]:.2f})',fontsize=8)
    axes[row,3].text(0.1,0.6,f'融合RGB=({blen_rgb[0]:.2f},{blen_rgb[1]:.2f},{blen_rgb[2]:.2f})',fontsize=8)
    axes[row,3].text(0.1,0.4,f'融合色: {recipe}',fontsize=7)

plt.tight_layout()
out='test_output/color_reconstruct.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'-> {out}')

print('\n原色 vs 融合色:')
for row,(name,img) in enumerate(imgs):
    x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad(): field=pipe(x)
    blended=blended_gua_response(field)
    o=img[0,0]; b=blended[32,32].astype(np.float32)/255
    print(f'{name:>4s}: 原=({o[0]:.2f},{o[1]:.2f},{o[2]:.2f}) -> 混=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f})')
