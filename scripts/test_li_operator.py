"""单算子测试：离 — 验证亮度响应"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda'
from src.operators import FIXED_BAGUA_COLORS

# ─── 合成测试图 ───
# 灰度渐变: 左暗(0.1) → 右亮(0.9)，测试离的连续响应
gradient = np.zeros((224, 224, 3), dtype=np.float32)
for x in range(224):
    gradient[:, x] = [x/224, x/224, x/224]  # 灰度渐变

# 红渐变: 左黑 → 右红亮
red_grad = np.zeros((224, 224, 3), dtype=np.float32)
for x in range(224):
    red_grad[:, x] = [x/224, x*0.05/224, x*0.05/224]  # 红色渐变

# 蓝到红渐变: 左蓝 → 右红
blue2red = np.zeros((224, 224, 3), dtype=np.float32)
for x in range(224):
    t = x/224
    blue2red[:, x] = [t, 0.0, 1.0-t]  # 蓝→红

# 杯子图
cup = cv2.imread('test_maccup.png')
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup = cv2.resize(cup, (224, 224))

fig, axes = plt.subplots(5, 3, figsize=(12, 18))

for row, (img, name) in enumerate([
    (gradient, '灰渐变\n左暗→右亮'),
    (red_grad, '红渐变\n左黑→右红亮'),
    (blue2red, '蓝红渐变\n左蓝→右红'),
    (cup, '杯子\n蓝底白杯'),
    (np.ones((224,224,3))*0.8, '纯亮灰\nR=G=B=0.8'),
]):
    # 原图
    axes[row, 0].imshow(img.clip(0,1))
    axes[row, 0].set_title(name, fontsize=9); axes[row, 0].axis('off')

    x = torch.from_numpy(img.copy()).permute(2,0,1).float().unsqueeze(0).to(device)

    # 离的颜色投影 [R=1.0, G=0.1, B=0.1] → 对红色/亮色敏感
    w_li = torch.tensor(FIXED_BAGUA_COLORS['li'], device=device).view(1,3,1,1)
    color_proj = (x * w_li).sum(dim=1, keepdim=True).clamp(min=0)
    cp = color_proj[0,0].cpu().numpy()

    # 方案A: 离 = 颜色投影值 (直通)
    li_a = cp

    # 方案B: 离 = 颜色投影 × 局部亮度
    from src.operators import _box_filter
    local_bright = _box_filter(color_proj.float(), k=9)[0,0].cpu().numpy()
    li_b = cp * local_bright

    axes[row, 1].imshow(li_a, cmap='hot')
    axes[row, 1].set_title(f'离=颜色投影\navg={li_a.mean():.3f} max={li_a.max():.3f}', fontsize=9)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(li_b, cmap='hot')
    axes[row, 2].set_title(f'离=投影×局部亮\navg={li_b.mean():.3f} max={li_b.max():.3f}', fontsize=9)
    axes[row, 2].axis('off')

plt.tight_layout()
out = 'test_output/test_li_operator.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

# 打印渐变线上的值
print()
print('=== 灰渐变(左→右) 离响应 ===')
for name, img in [('灰', gradient), ('红', red_grad), ('蓝红', blue2red)]:
    x = torch.from_numpy(img.copy()).permute(2,0,1).float().unsqueeze(0).to(device)
    w_li = torch.tensor(FIXED_BAGUA_COLORS['li'], device=device).view(1,3,1,1)
    cp = (x * w_li).sum(dim=1, keepdim=True).clamp(min=0)[0,0,112]
    vals = cp[::22].cpu().tolist()  # 每隔10%采样
    print(f'{name:>6s}: {"  ".join(f"{v:.3f}" for v in vals[:11])}')
