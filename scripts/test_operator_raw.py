"""原始算子测试：只看8个固定算子对杯子的裸响应，不涉及任何训练"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
op_cn = {'qian':'乾','kun':'坤','zhen':'震','xun':'巽','kan':'坎','li':'离','gen':'艮','dui':'兑'}

device = 'cuda'
from src.operators import BASE_OPS, FIXED_BAGUA_COLORS, ColorFixedOperator

img = cv2.imread('test_maccup.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_224 = cv2.resize(img_rgb, (224, 224))
x = torch.from_numpy(img_224).permute(2,0,1).float().unsqueeze(0).to(device)/255

# 每个算子用自己的固定颜色投影
ops = {name: ColorFixedOperator(fn, FIXED_BAGUA_COLORS[name]).to(device)
       for name, fn in BASE_OPS.items()}

with torch.no_grad():
    raw = {name: ops[name](x) for name in BASE_OPS}

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes[0,0].imshow(img_224); axes[0,0].set_title('原图'); axes[0,0].axis('off')
axes[0,1].axis('off'); axes[0,2].axis('off'); axes[0,3].axis('off')

# 3 个关键 ROI
rois = [
    ('杯内中心', 80, 140, 80, 140),
    ('无把手侧\n杯壁', 155, 175, 80, 140),      # y, x: 杯右无把手侧
    ('有把手侧\n杯壁', 90, 110, 140, 180),      # y, x: 杯左有把手侧
    ('背景', 180, 200, 15, 50),
]

results = {}
for i, (name, fn) in enumerate(BASE_OPS.items()):
    row, col = divmod(i + 4, 4)  # 从第2行开始
    ax = axes[row, col]
    v = raw[name][0, 0].cpu().numpy()
    vmin, vmax = np.percentile(v, 2), np.percentile(v, 98)
    ax.imshow(v, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f'{op_cn[name]}({name}) peak={v.max():.0f}', fontsize=9)
    ax.axis('off')
    results[name] = {}
    for rname, y1, y2, x1, x2 in rois:
        results[name][rname] = v[y1:y2, x1:x2].mean()

plt.tight_layout()
plt.savefig('test_output/operator_raw_baseline.png', dpi=120)
plt.close()
print('→ test_output/operator_raw_baseline.png')

# 打印表格
print()
print(f'{"":>12s}', end='')
for rname, *_ in rois:
    print(f'  {rname:>10s}', end='')
print()
print('-' * 56)
for name in BASE_OPS:
    print(f'  {op_cn[name]:>8s}  ', end='')
    for rname, *_ in rois:
        v = results[name][rname]
        print(f'  {v:>10.1f}', end='')
    print()

# 检查哪些算子在无把手侧有响应
print()
print('=== 无把手侧杯壁检测能力（杯内/背景比）===')
for name in BASE_OPS:
    nha = results[name]['无把手侧\n杯壁']
    bg = results[name]['背景']
    ratio = nha / (bg + 1e-6)
    status = '✓' if ratio > 1.2 else ('~' if ratio > 0.9 else '✗')
    print(f'  {op_cn[name]:>6s}: wall={nha:.1f}  bg={bg:.1f}  ratio={ratio:.1f}x  {status}')
