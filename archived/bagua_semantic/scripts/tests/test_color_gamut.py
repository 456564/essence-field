"""8卦色域: 8个固定卦色能否混合出目标色"""
import numpy as np

# 8卦固定颜色
colors=np.array([[0.9,0.3,0.15],[0.15,0.65,0.25],[0.1,0.55,0.85],[0.85,0.75,0.25],
                 [0.0,0.25,0.65],[0.8,0.05,0.05],[0.95,0.55,0.15],[0.35,0.65,0.9]])

# 目标色
targets={'赤':(1,0,0),'绿':(0,1,0),'蓝':(0,0,1),'白':(1,1,1),'黄':(1,1,0),
         '青':(0,1,1),'紫':(1,0,1),'黑':(0,0,0),'棕':(0.5,0.2,0),'灰蓝':(0.2,0.5,0.8)}

from scipy.optimize import nnls

print('8卦色 → 目标色 重建误差')
print(f'{"目标":>6s}  {"原RGB":>14s}  {"混合RGB":>14s}  {"误差":>6s}  {"主卦(权重)":>30s}')
print('-'*78)
for name,(r,g,b) in targets.items():
    target=np.array([r,g,b])
    w,_=nnls(colors.T,target)  # 非负最小二乘: 权重≥0
    w=w/(w.sum()+1e-6)  # 归一化
    blended=w@colors
    err=np.abs(blended-target).mean()
    top=np.argsort(-w)[:3]
    opn=['乾','坤','震','巽','坎','离','艮','兑']
    recipe=' '.join(f'{opn[t]}={w[t]*100:.0f}%' for t in top)
    print(f'{name:>6s}  ({r:.1f},{g:.1f},{b:.1f})  ({blended[0]:.2f},{blended[1]:.2f},{blended[2]:.2f})  {err:.3f}  {recipe}')

print()
print('→ 误差<0.1=可精确混合, 0.1-0.2=近似, >0.2=色域外')
