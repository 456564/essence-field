"""新算子联合验证：离(亮度)、坤(平坦)、坎(容器) + 震(不动)"""
import sys; sys.path.insert(0,'.')
import torch, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda'
from src.operators import BASE_OPS, FIXED_BAGUA_COLORS, ColorFixedOperator

img = cv2.imread('test_maccup.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
img = cv2.resize(img_rgb, (224, 224))
x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

ops = {n: ColorFixedOperator(fn, FIXED_BAGUA_COLORS[n]).to(device)
       for n, fn in BASE_OPS.items()}

with torch.no_grad():
    raw = {n: ops[n](x)[0,0].cpu().numpy() for n in BASE_OPS}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
targets = ['qian','kun','zhen','xun','kan','li','gen','dui']
op_cn = {'qian':'乾','kun':'坤','zhen':'震','xun':'巽','kan':'坎','li':'离','gen':'艮','dui':'兑'}
expected = {'kun':'平坦区亮','kan':'杯内部亮','li':'亮区亮','zhen':'边缘亮'}

for i, name in enumerate(targets):
    row, col = divmod(i, 4)
    v = raw[name]
    ax = axes[row, col]
    p2, p98 = np.percentile(v, 2), np.percentile(v, 98)
    ax.imshow(v, cmap='hot', vmin=p2, vmax=p98)
    title = f'{op_cn[name]}({name}) peak={v.max():.0f}'
    if name in expected: title += f'\n预期: {expected[name]}'
    ax.set_title(title, fontsize=9); ax.axis('off')

plt.tight_layout()
out = 'test_output/operators_v2_test.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

print()
for name, exp in expected.items():
    v = raw[name]
    cup_c = v[80:140, 70:140].mean()
    bg = v[180:, 20:60].mean()
    ok = '✓' if cup_c > bg else '✗'
    print(f'{op_cn[name]:>6s}: 杯内={cup_c:.3f}  bg={bg:.3f}  {ok}  (预期: {exp})')
