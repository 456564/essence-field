"""追溯单个像素通过8算子的完整数据流"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np

device = 'cuda'
img = cv2.imread('test_maccup.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_224 = cv2.resize(img_rgb, (224, 224))
x = torch.from_numpy(img_224).permute(2,0,1).float().unsqueeze(0).to(device)/255

from src.operators import BASE_OPS, FIXED_BAGUA_COLORS, ColorFixedOperator

# 3 个像素的坐标 (224x224)
pixels = {
    '杯内中心': (112, 112),    # cup interior center
    '杯壁/边缘': (80, 110),    # cup wall edge
    '椅背': (180, 112),        # background
}

op_cn = {'qian':'乾','kun':'坤','zhen':'震','xun':'巽','kan':'坎','li':'离','gen':'艮','dui':'兑'}
ops = {name: ColorFixedOperator(fn, FIXED_BAGUA_COLORS[name]).to(device)
       for name, fn in BASE_OPS.items()}

with torch.no_grad():
    raw = {name: ops[name](x) for name in BASE_OPS}
    # 颜色投影之前的值
    for name in BASE_OPS:
        w = torch.tensor(FIXED_BAGUA_COLORS[name], dtype=torch.float32).view(1,3,1,1).to(device)
        color_proj = (x * w).sum(dim=1, keepdim=True).clamp(min=0)
        # store
        raw[name] = {'color_proj': color_proj, 'operator_out': ops[name](x)}

print('=' * 90)
print('单个像素追踪：RGB → 颜色投影 → 算子输出')
print('=' * 90)

# 1. RGB 值
print()
print('【第0步】原始 RGB (范围 0~1)')
print(f'{"像素":>12s}: {"R":>6s} {"G":>6s} {"B":>6s}')
for pname, (y, x_pos) in pixels.items():
    r, g, b = x[0, :, y, x_pos].cpu().tolist()
    print(f'{pname:>12s}: {r:6.3f} {g:6.3f} {b:6.3f}')

# 2. 颜色投影 (每个像素 8 个值，来自 8 个颜色权重)
print()
print('【第1步】颜色投影 (RGB × 卦颜色权重 → 1通道响应)')
print(f'{"卦":>6s} {"权重[R,G,B]":>18s} {"颜色范围":>10s}', end='')
for pname in pixels: print(f'  {pname:>12s}', end='')
print()
for name in BASE_OPS:
    w = FIXED_BAGUA_COLORS[name]
    print(f'{op_cn[name]:>6s} [{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}] {"":>8s}', end='')
    for pname, (y, x_pos) in pixels.items():
        v = (x[0, :, y, x_pos] * torch.tensor(w, device=device)).sum().clamp(min=0).item()
        print(f'  {v:>12.3f}', end='')
    print()

# 3. 算子输出 (每个像素 8 个值)
print()
print('【第2步】算子输出 (颜色投影 → 几何算子 → 标量响应)')
print(f'{"卦":>6s}  {"算子":>10s}', end='')
for pname in pixels: print(f'  {pname:>12s}', end='')
print()
for name in BASE_OPS:
    v = raw[name]
    print(f'{op_cn[name]:>6s}  {"容器/方差等":>10s}', end='')
    for pname, (y, x_pos) in pixels.items():
        val = v[y, x_pos].item() if isinstance(v, torch.Tensor) else 0
        print(f'  {val:>12.3f}', end='')
    print()

# 4. 算子输出后 → /amax 归一化
print()
print('【第3步】/amax 归一化后 (除以该算子全局最大值)')
normed = {}
for name in BASE_OPS:
    out = ops[name](x)
    normed[name] = out / (out.amax() + 1e-6)

for name in BASE_OPS:
    print(f'{op_cn[name]:>6s}', end='')
    for pname, (y, x_pos) in pixels.items():
        val = normed[name][0, 0, y, x_pos].item()
        print(f'  {val:>12.3f}', end='')
    print()

# 5. 投影后 (1×1 conv, 1→8) × 8权重 → 8维向量
print()
print('【第4步】投影层 (1×1 conv × 8权重, 1→8维)')
import glob
from src.pipeline import BaguaPipeline
pipe = BaguaPipeline().to(device).eval()
ckpt_path = sorted(glob.glob('checkpoints_fixedcolor/bootstrap_epoch*.pth'))[-1]
print(f'  权重来源: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location=device)
pipe.fusion.W_up.data = ckpt.get("W_up", ckpt.get("A")); pipe.fusion.W_dn.data = ckpt.get("W_dn", ckpt.get("A"))
pipe.operator_layer.projections.load_state_dict(ckpt['proj'])

with torch.no_grad():
    full_field = pipe(x)

for name in BASE_OPS:
    # 获取该算子的投影权重[8] 和归一化值
    w = pipe.operator_layer.projections[name].weight.data.flatten()
    norm_val = normed[name]
    has_name = False
    for ch in range(min(3, 8)):  # 只看前3个通道
        if w[ch].item() > 1e-4:  # 只显示有意义的通道
            if not has_name:
                print(f'{op_cn[name]:>6s}', end='')
                has_name = True
            else:
                print(f'      ', end='')
            print(f'  ch{ch}: w={w[ch].item():+.4f} ', end='')
            for pname, (y, x_pos) in pixels.items():
                pv = norm_val[0, 0, y, x_pos].item()
                proj_val = w[ch].item() * pv
                print(f'  {pname}={proj_val:>+.4f}', end='')
            print()

# 6. 最终 64 维场中该像素的卦强度
print()
print('【第5步】64维场中的卦强度 (8个卦块的 L2 norm)')
op_names = list(BASE_OPS.keys())
for i, name in enumerate(op_names):
    block = full_field[0, i*8:(i+1)*8, :, :]
    strength = block.norm(dim=0)
    print(f'{op_cn[name]:>6s}', end='')
    for pname, (y, x_pos) in pixels.items():
        val = strength[y, x_pos].item()
        print(f'  {val:>12.3f}', end='')
    print()
