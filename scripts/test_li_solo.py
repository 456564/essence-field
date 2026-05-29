"""离算子单独测试：数据 + 可视化"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

# ─── 合成测试图 ───
H, W = 64, 256
imgs = {}

# 1. 纯色条: 红 | 绿 | 蓝 | 白 | 黄 | 青 | 紫 | 黑
strips = np.zeros((H, W, 3), dtype=np.float32)
for i, (r,g,b,name) in enumerate([(1,0,0,'红'),(0,1,0,'绿'),(0,0,1,'蓝'),(1,1,1,'白'),
                                     (1,1,0,'黄'),(0,1,1,'青'),(1,0,1,'紫'),(0,0,0,'黑')]):
    x0 = i * W//8; x1 = (i+1) * W//8
    strips[:, x0:x1] = [r, g, b]
imgs['1_纯色条'] = strips

# 2. 红渐变: 左黑→右纯红
grad = np.zeros((H, W, 3), dtype=np.float32)
for x in range(W):
    t = x/(W-1)
    grad[:, x] = [t, 0, 0]
imgs['2_红渐变'] = grad

# 3. 红→白渐变: 左红→右白
r2w = np.zeros((H, W, 3), dtype=np.float32)
for x in range(W):
    t = x/(W-1)
    r2w[:, x] = [1.0, t, t]  # R=1 constant, G+B go 0→1
imgs['3_红→白'] = r2w

# 4. 灰渐变: 左黑→右白
grey = np.zeros((H, W, 3), dtype=np.float32)
for x in range(W):
    t = x/(W-1)
    grey[:, x] = [t, t, t]
imgs['4_灰渐变'] = grey

# 5. 红灰混合: 左红→右灰
r2g = np.zeros((H, W, 3), dtype=np.float32)
for x in range(W):
    t = x/(W-1)
    r2g[:, x] = [1-t*0.5, t*0.5, t*0.5]  # R 1→0.5, G+B 0→0.5
imgs['5_红→灰'] = r2g

# ─── 离算子 = R - 0.5*(G+B), clamp≥0 ───
def li_operator(rgb):
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return np.clip(R - 0.5*(G+B), 0, None)

# ─── 杯子图 ───
import cv2
cup = cv2.imread('test_maccup.png')
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup = cv2.resize(cup, (224, 224))
imgs['6_杯子'] = cup

# ─── 可视化 ───
fig, axes = plt.subplots(len(imgs), 3, figsize=(14, 2.5*len(imgs)))

for row, (title, img) in enumerate(imgs.items()):
    # 原图
    axes[row,0].imshow(img.clip(0,1))
    axes[row,0].set_title(f'{title} 原图', fontsize=9); axes[row,0].axis('off')

    # 离响应热力图
    li = li_operator(img)
    axes[row,1].imshow(li, cmap='hot', vmin=0, vmax=max(li.max(), 1e-6))
    axes[row,1].set_title(f'离 热力图\nmin={li.min():.3f} max={li.max():.3f} avg={li.mean():.3f}', fontsize=9)
    axes[row,1].axis('off')

    # 中线剖面
    mid_y = img.shape[0]//2
    profile = li[mid_y, :]
    axes[row,2].plot(profile, 'r-', linewidth=0.8)
    axes[row,2].set_ylim(0, max(li.max(), 1e-6)*1.1)
    axes[row,2].set_title(f'离值 中线剖面\n', fontsize=9)
    axes[row,2].set_xlabel('x像素')

plt.tight_layout()
out = 'test_output/test_li_solo.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

# ─── 数值报告 ───
print()
print('='*70)
print('离算子数值报告:  out = max(R - 0.5*(G+B), 0)')
print('='*70)

# 纯色条
for i,(r,g,b,name) in enumerate([(1,0,0,'红'),(0,1,0,'绿'),(0,0,1,'蓝'),(1,1,1,'白'),
                                   (1,1,0,'黄'),(0,1,1,'青'),(1,0,1,'紫'),(0,0,0,'黑')]):
    v = li_operator(np.array([[[r,g,b]]]))[0,0]
    print(f'  {name:>4s} RGB=[{r},{g},{b}] → 离={v:.2f}')

# 杯子关键区域
cup_li = li_operator(cup)
cup_body = cup_li[80:140, 70:140].mean()
cup_rim = cup_li[55:70, 70:150].mean()
cup_shadow = cup_li[140:180, 140:180].mean()
bg = cup_li[180:, 20:60].mean()
print(f'\n  杯子图:')
print(f'    杯身(白)={cup_body:.4f}  杯口高光={cup_rim:.4f}  阴影={cup_shadow:.4f}  背景(蓝灰)={bg:.4f}')
print(f'    杯身>背景? {"YES" if cup_body>bg else "NO"} | 预期: 白杯≈蓝灰≈0, 都低')
print(f'    杯口>杯身? {"YES" if cup_rim>cup_body else "NO"} | 高光中R>G+B, 应有微高')
