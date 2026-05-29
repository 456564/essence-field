"""单算子投影测试：跳过A核融合，只看每个卦的1→8投影输出"""
import sys; sys.path.insert(0, '.')
import torch, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda'

# 加载模型和权重（只用到 projection 和 base_ops）
from src.pipeline import MultiDimOperatorLayer
from src.operators import BAGUA_OPERATORS
layer = MultiDimOperatorLayer().to(device).eval()
ckpt_path = sorted(glob.glob('checkpoints_fixedcolor/bootstrap_epoch*.pth'))[-1]
print(f'加载: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location=device)
layer.projections.load_state_dict(ckpt['proj'])

# 读图
img = cv2.imread('test_maccup.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_small = cv2.resize(img_rgb, (128, 128))
x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device)/255

with torch.no_grad():
    raw = layer.base_ops(x)  # 原始算子输出
    # 手动跑投影，跳过instance_norm（用/amax）
    proj_out = {}
    for name in BAGUA_OPERATORS:
        out = raw[name]
        out = out / (out.amax(dim=(2,3), keepdim=True) + 1e-6)
        feat = layer.projections[name](out)  # [B, 8, H, W]
        proj_out[name] = feat

# ─── 可视化：每行 = 一个算子，4列 = 原始 / 投影norm / 投影ch0 / 投影ch1 ───
op_names = list(BAGUA_OPERATORS)
fig, axes = plt.subplots(8, 4, figsize=(14, 24))

for row, name in enumerate(op_names):
    raw_map = raw[name][0].squeeze().cpu().numpy()
    proj_norm = proj_out[name][0].norm(dim=0).cpu().numpy()
    ch0 = proj_out[name][0, 0].cpu().numpy()
    ch1 = proj_out[name][0, 1].cpu().numpy()

    # 原始算子输出
    ax = axes[row, 0]
    im = ax.imshow(raw_map, cmap='hot')
    cup_v = raw_map[40:88, 40:88].mean()
    bg_v = raw_map[80:, :30].mean()
    ax.set_title(f'{name} raw\n杯内={cup_v:.1f} bg={bg_v:.1f}', fontsize=8)
    ax.axis('off')

    # 投影后 8 维 L2 范数
    ax = axes[row, 1]
    ax.imshow(proj_norm, cmap='hot')
    pc = proj_norm[40:88, 40:88].mean()
    pb = proj_norm[80:, :30].mean()
    ok = '✓' if pc > pb else '✗'
    ax.set_title(f'{name} |proj|_2\n杯内={pc:.3f} bg={pb:.3f} {ok}', fontsize=8)
    ax.axis('off')

    # 投影 ch0
    ax = axes[row, 2]
    c0_c = ch0[40:88, 40:88].mean()
    c0_b = ch0[80:, :30].mean()
    ax.imshow(ch0, cmap='RdBu_r', vmin=-abs(ch0).max(), vmax=abs(ch0).max())
    ax.set_title(f'ch0 杯内={c0_c:.3f} bg={c0_b:.3f}', fontsize=8)
    ax.axis('off')

    # 投影 ch1
    ax = axes[row, 3]
    c1_c = ch1[40:88, 40:88].mean()
    c1_b = ch1[80:, :30].mean()
    ax.imshow(ch1, cmap='RdBu_r', vmin=-abs(ch1).max(), vmax=abs(ch1).max())
    ax.set_title(f'ch1 杯内={c1_c:.3f} bg={c1_b:.3f}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
out_path = 'test_output/projection_alone.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'→ {out_path}')

# 关键：检查坤的投影后极性
print()
print('=== 坤投影极性 ===')
kw = layer.projections['kun'].weight.data.flatten().tolist()
print(f'坤投影权重: {[round(w,4) for w in kw]}')
proj_n = proj_out['kun'][0].norm(dim=0).cpu().numpy()
print(f'投影后范数: 杯内={proj_n[40:88,40:88].mean():.3f}  bg={proj_n[80:,:30].mean():.3f}')
print(f'杯内>背景? {"✓ 正确" if proj_n[40:88,40:88].mean() > proj_n[80:,:30].mean() else "✗ 反转"}')
