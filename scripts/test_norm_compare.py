"""对比 4 种归一化方案在卦复合图上的效果"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.pipeline import BaguaPipeline, BAGUA_OPERATORS
from src.visualize import blended_gua_response, GUA_COLORS
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda'
model = BaguaPipeline().to(device).eval()

# 读杯子图
img = cv2.imread('test_maccup.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_small = cv2.resize(img_rgb, (128, 128))
x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device)/255

# 获取原始算子输出
with torch.no_grad():
    base = model.operator_layer.base_ops(x)

# 收集每个算子的原始值
op_names = list(BAGUA_OPERATORS.keys())
raw = {name: base[name] for name in op_names}

# 4 种归一化
norms = {}

# 方案 A: 当前 /amax
norms['A: /amax'] = {}
for name in op_names:
    out = raw[name]
    norms['A: /amax'][name] = out / (out.amax(dim=(2,3), keepdim=True) + 1e-6)

# 方案 B: log1p 压缩
norms['B: log1p'] = {}
for name in op_names:
    out = raw[name]
    norms['B: log1p'][name] = torch.log1p(out)

# 方案 C: 全局 /max (所有算子共用一个分母)
all_maxes = torch.stack([raw[n].amax(dim=(2,3), keepdim=True) for n in op_names])
global_max = all_maxes.amax(dim=0, keepdim=True)  # 最大的算子峰值
norms['C: global /max'] = {}
for name in op_names:
    norms['C: global /max'][name] = raw[name] / (global_max + 1e-6)

# 方案 D: /95分位值
norms['D: /q95'] = {}
for name in op_names:
    out = raw[name]
    q95 = out.quantile(0.95)
    norms['D: /q95'][name] = out / (q95 + 1e-6)

# ─── 可视化 ───
fig, axes = plt.subplots(4, 9, figsize=(20, 12))

for row, (norm_name, norm_data) in enumerate(norms.items()):
    # 前 8 列：每个算子的热力图（归一化后）
    for col, name in enumerate(op_names):
        v = norm_data[name][0].squeeze().cpu().numpy()
        ax = axes[row, col]
        ax.imshow(v, cmap='hot', vmin=v.min(), vmax=v.max())
        mn, mx = v.mean(), v.max()
        ax.set_title(f'{name} clip={mx:.0f} avg={mn:.3f}', fontsize=8)
        ax.axis('off')

    # 第 9 列：blended 复合图（用归一化后的值模拟 field 中该卦块）
    rgba = np.zeros((128, 128, 3), dtype=np.float32)
    colors = np.array([plt.cm.colors.hex2color(c) for c in GUA_COLORS])
    for i, name in enumerate(op_names):
        v = norm_data[name][0].squeeze().cpu().numpy()
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        spread = hi - lo
        nd = np.clip((v - lo) / (spread + 1e-6), 0, 1)
        rgba += nd[:,:,np.newaxis] * colors[i]
    rgba = np.clip(rgba, 0, 1)
    axes[row, 8].imshow(rgba)
    axes[row, 8].set_title(f'{norm_name}\nblended', fontsize=9)
    axes[row, 8].axis('off')

plt.tight_layout()
out_path = 'test_output/norm_compare.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out_path}')

# 打印每个方案的对比度指标
print()
print(f'{"方案":20s} {"乾":>8s} {"坤":>8s} {"震":>8s} {"离":>8s} {"杯内坤/背景"}')
print('-' * 70)
for norm_name, norm_data in norms.items():
    vals = []
    for name in op_names:
        v = norm_data[name][0].squeeze().cpu().numpy()
        vals.append(v.max())
    # 坤的杯内vs背景
    kv = norm_data['kun'][0].squeeze().cpu().numpy()
    cup = kv[40:88, 40:88].mean()
    bg = kv[80:, :30].mean()
    ratio = cup/(bg+1e-6)
    s = f'{norm_name:20s}'
    for i, name in enumerate(op_names):
        s += f' {vals[i]:>8.1f}'
    s += f' {ratio:>8.2f}'
    print(s)
