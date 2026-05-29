"""坤算子诊断：对比原始输出 vs 投影后输出"""
import sys; sys.path.insert(0, '.')
from src.pipeline import BaguaPipeline
import torch, cv2, numpy as np, matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = 'cuda'
pipe = BaguaPipeline(d=8).to(device).eval()
ckpt = torch.load('checkpoints_fixedcolor/bootstrap_epoch15.pth', map_location=device)
pipe.fusion.A.data = ckpt['A']
pipe.operator_layer.projections.load_state_dict(ckpt['proj'])

img = cv2.imread('test_maccup.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
SIZE = 224
img_resized = cv2.resize(img_rgb, (SIZE, SIZE))
x = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).to(device)/255

with torch.no_grad():
    ops_out = pipe.operator_layer.base_ops(x)
    kun_raw = ops_out['kun'][0, 0].cpu().numpy()
    field = pipe(x)
    kun_in_field = field[0, 0:8, :, :].norm(dim=0).cpu().numpy()  # 坤的8维范数

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(kun_raw, cmap='hot', vmin=np.percentile(kun_raw, 2), vmax=np.percentile(kun_raw, 98))
axes[0].set_title('坤原始响应 (算子直接输出)')
axes[1].imshow(kun_in_field, cmap='hot', vmin=np.percentile(kun_in_field, 2), vmax=np.percentile(kun_in_field, 98))
axes[1].set_title('坤在64维场中的范数 (训练后投影)')
axes[2].imshow(img_resized)
axes[2].set_title('原图')

# 对比所有8卦在算子层的响应
axes[3].axis('off')

plt.tight_layout()
plt.savefig('test_output/kun_diagnostic.png', dpi=150)
plt.close()
print('诊断图 → test_output/kun_diagnostic.png')
print(f'kun_raw 杯身中心均值: {kun_raw[80:140, 80:140].mean():.3f}')
print(f'kun_raw 背景均值:       {kun_raw[kun_raw.shape[0]//2:, :40].mean():.3f}')
print(f'kun_in_field 杯身中心:  {kun_in_field[80:140, 80:140].mean():.3f}')
print(f'kun_in_field 背景:      {kun_in_field[kun_in_field.shape[0]//2:, :40].mean():.3f}')
