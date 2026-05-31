"""test_yibao.jpg 量化分析 — 不看图只看数"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np
from src.operators import PhysicalOperatorLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
op_layer = PhysicalOperatorLayer()

img = cv2.imread('test_yibao.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
x = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

with torch.no_grad():
    base = op_layer(x)

d, g, c, r, j, t, yg, yn, vp = [base[0,i].cpu().numpy() for i in range(9)]

names = ['dong(梯度)', 'gang(边界)', 'cu(纹理)', 'rou(渐变)',
         'ju(围合)', 'dist(距边)', 'yang(实体)', 'yin(虚空)', 'void_prob']
maps = [d, g, c, r, j, t, yg, yn, vp]

print(f"{'算子':<16} {'均值':>8} {'中位':>8} {'最大':>8} {'>0.5占比':>8}")
print("-"*56)
for name, m in zip(names, maps):
    above = (m > 0.5).mean() * 100
    print(f"{name:<16} {m.mean():>8.4f} {np.median(m):>8.4f} {m.max():>8.4f} {above:>7.1f}%")

# 关键指标
print(f"\n{'='*50}")
print("关键判据:")
print(f"  ju>0.5 + void>0.5 同时成立: {((j>0.5) & (vp>0.5)).mean()*100:.1f}%  ← 容器/空腔")
print(f"  yang>0.5 + ju>0.5 同时成立: {((yg>0.5) & (j>0.5)).mean()*100:.1f}%  ← 实体围合")
print(f"  yang>0.5 (实体覆盖): {(yg>0.5).mean()*100:.1f}%")
print(f"  void>0.5 (虚空): {(vp>0.5).mean()*100:.1f}%")
