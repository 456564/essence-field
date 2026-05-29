"""坤算子多图测试：不同平坦度的真实图片"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2, glob
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

DATA = 'data/caltech101/101_ObjectCategories'
W = 'data/caltech101/101_ObjectCategories'

# 挑选纹理差异大的类别
from scipy.ndimage import uniform_filter as bf

def kun_ch(img):
    # 坤 = 1/(1+5*mean(|laplacian|)) — 结构平坦度
    ch = img[:,:,0]  # 灰度即可
    from scipy.ndimage import laplace
    lap = np.abs(laplace(ch))
    ld = bf(lap, 31, mode='reflect')
    return 1.0/(ld*5.0 + 1.0)

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img, (224,224))

pairs = []
# 找纹理跨度大的类别：天空(无)、人面(有)、汽车(光滑)、砖墙(无)
for cat_name in ['BACKGROUND_Google', 'faces', 'car_side', 'bonsai', 'butterfly',
                 'chair', 'ketch', 'lotus', 'sunflower', 'watch']:
    files = sorted(glob.glob(f'{DATA}/{cat_name}/*.jpg'))
    if files:
        pairs.append((cat_name, load_img(files[len(files)//2])))

# 加杯子
cup = load_img('test_maccup.png')
pairs.insert(0, ('杯子', cup))

rows = len(pairs)
fig, axes = plt.subplots(rows, 3, figsize=(12, 2.5*rows))

for row, (name, img) in enumerate(pairs):
    axes[row,0].imshow(img)
    axes[row,0].set_title(name, fontsize=9); axes[row,0].axis('off')

    kun = kun_ch(img)
    axes[row,1].imshow(kun, cmap='hot', vmin=0, vmax=1)
    axes[row,1].set_title(f'坤 avg={kun.mean():.3f} max={kun.max():.3f}', fontsize=9)
    axes[row,1].axis('off')

    # 第3列: laplacian密度图 (边缘密度)
    from scipy.ndimage import laplace
    lap = np.abs(laplace(img[:,:,0]))
    ld = bf(lap, 31, mode='reflect')
    axes[row,2].imshow(ld, cmap='magma')
    axes[row,2].set_title(f'Laplacian密度\navg={ld.mean():.4f}', fontsize=9)
    axes[row,2].axis('off')

plt.tight_layout()
out='test_output/test_kun_multi.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

print()
print(f'{"图片":>12s}  {"坤avg":>8s}  {"坤max":>8s}  {"方差avg":>8s}  判断')
print('-'*52)
for name, img in pairs:
    kun = kun_ch(img)
    desc = '平坦' if kun.mean()>0.8 else ('中等' if kun.mean()>0.5 else '纹理多')
    print(f'{name:>12s}  {kun.mean():>8.3f}  {kun.max():>8.3f}  {"-":>8s}  {desc}')
