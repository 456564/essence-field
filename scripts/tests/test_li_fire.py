"""离算子验证：在应响应的图上测试"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2, glob
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

def li_op(rgb):
    R,G,B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return np.clip(R - 0.5*(G+B), 0, None)

# ─── 合成测试 ───
H, W = 128, 128
synth = {}

# 红圆蓝底
img = np.zeros((H, W, 3), dtype=np.float32); img[:,:] = [0.2, 0.3, 0.8]
yy, xx = np.ogrid[:H, :W]; r = np.sqrt((yy-64)**2+(xx-64)**2)
img[r<30] = [0.9, 0.05, 0.05]  # 红圆
synth['红圆蓝底'] = img

# 红绿渐变
img2 = np.zeros((H, W, 3), dtype=np.float32)
for x in range(W):
    t = x/(W-1)
    img2[:, x] = [1-t, t, 0]  # 红→绿
synth['红绿渐变'] = img2

# 真实照片：找 caltech101 里可能的红色物体
DATA = 'data/caltech101/101_ObjectCategories'
real = {}
for cat, n in [('butterfly',2), ('crab',1), ('crayfish',1)]:
    files = sorted(glob.glob(f'{DATA}/{cat}/*.jpg'))
    if len(files) > n:
        img = cv2.imread(files[n]); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        img = cv2.resize(img, (224,224))
        real[f'{cat}'] = img

# ─── 可视化 ───
all_imgs = {**synth, **real}
fig, axes = plt.subplots(len(all_imgs), 3, figsize=(12, 3*len(all_imgs)))

for row, (title, img) in enumerate(all_imgs.items()):
    axes[row,0].imshow(img.clip(0,1))
    axes[row,0].set_title(title, fontsize=10); axes[row,0].axis('off')

    li = li_op(img)
    axes[row,1].imshow(li, cmap='hot', vmin=0, vmax=max(li.max(),0.01))
    R,G,B = img.mean(axis=(0,1))
    axes[row,1].set_title(f'离 热力图 (avg={li.mean():.3f})', fontsize=9); axes[row,1].axis('off')

    # R,G,B 值和离值分布
    axes[row,2].axis('off')
    axes[row,2].text(0.1, 0.8, f'Avg RGB = [{R:.3f}, {G:.3f}, {B:.3f}]', fontsize=9)
    axes[row,2].text(0.1, 0.6, f'R-0.5(G+B) = {R-0.5*(G+B):.3f}', fontsize=9)
    axes[row,2].text(0.1, 0.4, f'离 max = {li.max():.3f}', fontsize=9)
    axes[row,2].text(0.1, 0.2, f'离>0 像素占比 = {(li>0.01).mean()*100:.1f}%', fontsize=9)

plt.tight_layout()
out='test_output/test_li_fire.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

# 杯子图单独对比
cup = cv2.imread('test_maccup.png')
cup = cv2.cvtColor(cup, cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup = cv2.resize(cup, (224,224))
cup_li = li_op(cup)
print(f'\n杯子: 离 avg={cup_li.mean():.4f}  max={cup_li.max():.4f}  >0占比={(cup_li>0.01).mean()*100:.1f}%')
print('杯子是白色陶瓷——几乎没离响应，这是对的。白色≠火。')

for title, img in all_imgs.items():
    li = li_op(img)
    print(f'{title:>10s}: 离 avg={li.mean():.4f}  max={li.max():.4f}  >0占比={(li>0.01).mean()*100:.1f}%')
