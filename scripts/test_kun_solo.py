"""坤算子单独测试：平坦/均质度 = 1/(1 + 20*local_var)"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

H, W = 128, 256
rng = np.random.RandomState(42)
imgs = {}

# 1. 纯色渐变: 左纯白→右纯白 (全平坦)
grad = np.ones((H,W,3), dtype=np.float32)*0.7
for x in range(W):
    grad[:,x] = [0.3 + 0.4*x/(W-1)]*3
imgs['1_纯色渐变\n(全平坦)'] = grad

# 2. 纯色 → 噪声渐变: 左平坦→右高频
flat2noise = np.ones((H,W,3), dtype=np.float32)*0.5
for x in range(W):
    t = x/(W-1)
    noise = rng.randn(H, 3) * t * 0.3
    flat2noise[:,x] = np.clip(0.5 + noise, 0, 1)
imgs['2_平坦→噪声\n(左均右噪)'] = flat2noise

# 3. 棋盘格 (纹理)
checker = np.zeros((H,W,3), dtype=np.float32)
for y in range(0,H,16):
    for x in range(0,W,16):
        v = 0.3 if ((y//16)+(x//16))%2==0 else 0.7
        checker[y:y+16, x:x+16] = v
imgs['3_棋盘格\n(高纹理)'] = checker

# 4. 平滑圆
circle = np.ones((H,W,3), dtype=np.float32)*0.3
yy,xx=np.ogrid[:H,:W]
r=np.sqrt((yy-64)**2+(xx-128)**2)
circle[r<40]=0.7
imgs['4_光滑圆\n(内部平坦)'] = circle

# 5. 杯子
cup = cv2.imread('test_maccup.png')
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup = cv2.resize(cup, (224,224))
imgs['5_杯子'] = cup

# ─── 坤算子 = 1/(1+20*local_var) ───
def box_filter(ch, k=7):
    from scipy.ndimage import uniform_filter
    return uniform_filter(ch.astype(np.float64), size=k, axes=(0,1), mode='reflect').astype(np.float32)

def kun_operator(rgb, color_weight):
    ch = (rgb * np.array(color_weight)).sum(axis=2, keepdims=True).clip(0)
    ch = ch[:,:,0]
    lm = box_filter(ch, k=31)
    lv = box_filter((ch-lm)**2, k=31)
    return 1.0/(lv*20.0 + 1.0)

# ─── 坤颜色权重 = 黑/土/暗 ───
KUN_W = [0.2, 0.5, 0.1]  # 归一化后 ≈ [0.365, 0.913, 0.183]
norm = np.sqrt(sum(w*w for w in KUN_W))
KUN_W_N = [w/norm for w in KUN_W]

fig, axes = plt.subplots(len(imgs), 3, figsize=(14, 2.8*len(imgs)))

for row, (title, img) in enumerate(imgs.items()):
    axes[row,0].imshow(img.clip(0,1))
    axes[row,0].set_title(title, fontsize=9); axes[row,0].axis('off')

    kun = kun_operator(img, KUN_W_N)
    axes[row,1].imshow(kun, cmap='hot', vmin=0, vmax=1)
    axes[row,1].set_title(f'坤 热力图\nmin={kun.min():.3f} max={kun.max():.3f} avg={kun.mean():.3f}', fontsize=9)
    axes[row,1].axis('off')

    # 中线剖面
    mid = kun.shape[0]//2
    profile = kun[mid, :]
    axes[row,2].plot(profile, 'g-', linewidth=0.8)
    axes[row,2].set_ylim(0, 1)
    axes[row,2].set_title(f'中线剖面', fontsize=9)

plt.tight_layout()
out = 'test_output/test_kun_solo.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

# ─── 数值报告 ───
print()
print('='*70)
print(f'坤算子数值报告:  out = 1/(1+20*lv)  color=[{KUN_W_N[0]:.3f},{KUN_W_N[1]:.3f},{KUN_W_N[2]:.3f}]')
print('='*70)
for title, img in imgs.items():
    kun = kun_operator(img, KUN_W_N)
    print(f'{title:>20s}: 平坦=高  min={kun.min():.3f}  max={kun.max():.3f}')

# 杯子区域
cup_kun = kun_operator(cup, KUN_W_N)
print(f'\n杯子细节:')
for name, y1,y2,x1,x2 in [('杯内',80,140,70,140),('杯壁',80,140,140,160),('背景',180,224,20,80)]:
    v = cup_kun[y1:y2,x1:x2].mean()
    print(f'  {name}: {v:.4f}')
