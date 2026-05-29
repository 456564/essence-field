"""crab单独调试 — 为什么零训练分离失败"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline
from src.operators import BASE_OPS, FIXED_BAGUA_COLORS, ColorFixedOperator

pipe=BaguaPipeline().to(device).eval()
pipe.fusion.W_up.data=torch.eye(8,device=device)
pipe.fusion.W_dn.data=torch.eye(8,device=device)
for proj in pipe.operator_layer.projections.values(): proj.weight.data.fill_(1.0)

crab=cv2.imread('data/caltech101/101_ObjectCategories/crab/image_0001.jpg')
crab=cv2.cvtColor(crab,cv2.COLOR_BGR2RGB).astype(np.float32)/255
crab=cv2.resize(crab,(224,224))
x=torch.from_numpy(crab).permute(2,0,1).unsqueeze(0).to(device)

with torch.no_grad():
    field=pipe(x)
    raw={n: fn(crab[:,:,0]) for n,fn in [('qian',lambda _:0)]}  # dummy, 用实际算子
    ops_out={n: ColorFixedOperator(fn,FIXED_BAGUA_COLORS[n]).to(device)(x)[0,0].cpu().numpy()
             for n,fn in BASE_OPS.items()}

# 用实际算子输出
with torch.no_grad():
    ops_out2=pipe.operator_layer.base_ops(x)

norms=field[0].norm(dim=0).cpu().numpy()
median=np.median(norms)
mask=norms>median

opn=['qian','kun','zhen','xun','kan','li','gen','dui']
opc={'qian':'乾Q','kun':'坤K','zhen':'震ZH','xun':'巽X','kan':'坎KA','li':'离L','gen':'艮G','dui':'兑D'}

fig,axes=plt.subplots(3,4,figsize=(16,10))
axes[0,0].imshow(crab.clip(0,1)); axes[0,0].set_title('crab原图'); axes[0,0].axis('off')
axes[0,1].imshow(norms,cmap='hot'); axes[0,1].set_title(f'||field|| median={median:.0f}'); axes[0,1].axis('off')
axes[0,2].imshow(mask,cmap='gray'); axes[0,2].set_title(f'mask obj={mask.mean()*100:.0f}%'); axes[0,2].axis('off')
axes[0,3].axis('off')
axes[2,3].axis('off')

# 螃蟹主体区域 vs 背景区域的算子裸值
# 红壳区域
shell_y,shell_x=60,100; shell_r=40
yy_mask=(np.arange(224)[:,None]-shell_y)**2+(np.arange(224)-shell_x)**2<shell_r**2
# 暗岩区域
rock_y,rock_x=190,30; rock_r=25
rock_mask=(np.arange(224)[:,None]-rock_y)**2+(np.arange(224)-rock_x)**2<rock_r**2

print('=== 原始算子输出(逐像素) ===')
print(f'{"算子":>6s}  {"红壳":>8s}  {"暗岩":>8s}  比值')
for i,name in enumerate(opn):
    raw=ops_out2[name][0,0].cpu().numpy()
    sv=raw[shell_y-shell_r:shell_y+shell_r,shell_x-shell_r:shell_x+shell_r].mean()
    rv=raw[rock_y-rock_r:rock_y+rock_r,rock_x-rock_r:rock_x+rock_r].mean()
    print(f'{opc[name]:>6s}  {sv:8.1f}  {rv:8.1f}  {sv/(rv+1e-6):6.1f}x')

# 64维场中各卦强度
print('\n=== 64维场中各卦强度 ===')
for i,name in enumerate(opn):
    v=field[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy()
    sv=v[shell_y-shell_r:shell_y+shell_r,shell_x-shell_r:shell_x+shell_r].mean()
    rv=v[rock_y-rock_r:rock_y+rock_r,rock_x-rock_r:rock_x+rock_r].mean()
    print(f'{opc[name]:>6s}  {sv:8.1f}  {rv:8.1f}  {sv/(rv+1e-6):6.1f}x')

# 可视化各算子
for i,name in enumerate(opn):
    row=i//4+1; col=i%4
    v=ops_out2[name][0,0].cpu().numpy()
    p2,p98=np.percentile(v,2),np.percentile(v,98)
    axes[row,col].imshow(v,cmap='hot',vmin=p2,vmax=p98)
    sv=v[yy_mask].mean(); rv=v[rock_mask].mean()
    axes[row,col].set_title(f'{opc[name]} 壳={sv:.0f}岩={rv:.0f}',fontsize=9)
    axes[row,col].axis('off')
axes[2,3].axis('off')

plt.tight_layout()
out='test_output/crab_debug.png'
plt.savefig(out,dpi=120,bbox_inches='tight')
plt.close()
print(f'\n-> {out}')
