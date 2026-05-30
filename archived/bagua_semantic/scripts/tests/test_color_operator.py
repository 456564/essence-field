"""颜色投影 + 算子联合验证"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'; H,W=128,128
from src.operators import BASE_OPS, FIXED_BAGUA_COLORS, ColorFixedOperator, BAGUA_NAMES

ops={n: ColorFixedOperator(fn, FIXED_BAGUA_COLORS[n]).to(device) for n,fn in BASE_OPS.items()}
op_names=list(BASE_OPS.keys())
op_cn={n:BAGUA_NAMES[n] for n in op_names}

# ─── 测试图 ───
imgs={}

# 1. 8色条
cols8=np.zeros((H,256,3),np.float32)
for i,(r,g,b,name) in enumerate([(1,0,0,'赤'),(0,1,0,'青'),(0,0,1,'蓝'),(1,1,1,'白'),
                                   (1,1,0,'黄'),(0,1,1,'青绿'),(1,0,1,'紫'),(0,0,0,'黑')]):
    cols8[:,i*32:(i+1)*32]=[r,g,b]
imgs['8色条']=cols8

# 2. 杯子
cup=cv2.imread('test_maccup.png')
cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224))
imgs['杯子']=cup

# 3. 红渐变(离测试)
rg=np.zeros((H,256,3),np.float32)
for x in range(256): rg[:,x]=[1.0, x/256, x/256]  # 红→白
imgs['红→白(离渐变)']=rg

# 4. 蓝渐变(坎测试)
bg=np.zeros((H,256,3),np.float32)
for x in range(256): bg[:,x]=[x/256, x/256, 1.0]  # 黑→蓝
imgs['黑→蓝(坎渐变)']=bg

# ─── 统一跑8算子 ───
for img_name, img in imgs.items():
    Hc,Wc=img.shape[:2]
    x=torch.from_numpy(img.copy().astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)

    fig,axes=plt.subplots(1,10,figsize=(22,2.5))
    axes[0].imshow(img.clip(0,1)); axes[0].set_title(img_name,fontsize=8); axes[0].axis('off')

    with torch.no_grad():
        raw={n: ops[n](x)[0,0].cpu().numpy() for n in op_names}

    for col, name in enumerate(op_names):
        v=raw[name]
        p2,p98=np.percentile(v,2),np.percentile(v,98)
        axes[col+1].imshow(v,cmap='hot',vmin=p2,vmax=p98)
        axes[col+1].set_title(f'{op_cn[name]}\navg={v.mean():.2f}',fontsize=7)
        axes[col+1].axis('off')
    # 空出1个: 8算子+原图=9列, 10个子图里有1个空, 把最后一个隐藏
    axes[9].axis('off')

    plt.tight_layout()
    out=f'test_output/color_op_{img_name}.png'
    plt.savefig(out,dpi=100,bbox_inches='tight')
    plt.close()
    print(f'→ {out}')

# ─── 8色条数值矩阵 ───
print()
cols8_names=['赤','青','蓝','白','黄','青绿','紫','黑']
x=torch.from_numpy(cols8.copy().astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
with torch.no_grad():
    raw8={n: ops[n](x)[0,0].cpu().numpy() for n in op_names}

print('='*80)
print('颜色响应矩阵: 每个算子对8种纯色的响应 (avg over each 32px stripe)')
print(f'{"":>6s}',end='')
for cn in cols8_names: print(f'  {cn:>6s}',end='')
print()
for name in op_names:
    print(f'{op_cn[name]:>6s}',end='')
    for i in range(8):
        v=raw8[name][:,i*32:(i+1)*32].mean()
        print(f'  {v:6.3f}',end='')
    print()
print()
print('预期: 乾→赤高, 坤→[绿/暗]高, 震→青绿高, 巽→白高, 坎→蓝高, 离→赤高, 艮→黄高, 兑→白/蓝高')
