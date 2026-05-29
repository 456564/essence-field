"""坤测试：草地 — 多色但无纹理 = 应该平坦"""
import sys; sys.path.insert(0,'.')
import numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

from scipy.ndimage import uniform_filter as bf, laplace

def kun_new(img):
    """新坤：Laplacian密度"""
    lap = np.abs(laplace(img[:,:,0]))
    ld = bf(lap, 31, mode='reflect')
    return 1.0/(ld*5.0 + 1.0)

def kun_old(img):
    """旧坤：颜色方差"""
    ch = img[:,:,0]
    lm = bf(ch, 31, mode='reflect')
    lv = bf((ch-lm)**2, 31, mode='reflect')
    return 1.0/(lv*20.0 + 1.0)

imgs = {}
for path, name in [('image.png','草地'),('test_maccup.png','杯子')]:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    imgs[name] = cv2.resize(img, (224,224))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, (name, img) in enumerate(imgs.items()):
    axes[row,0].imshow(img); axes[row,0].set_title(f'{name} 原图',fontsize=10)
    axes[row,0].axis('off')

    kn = kun_new(img)
    axes[row,1].imshow(kn, cmap='hot', vmin=0, vmax=1)
    axes[row,1].set_title(f'新坤(Laplacian) avg={kn.mean():.3f}',fontsize=9)
    axes[row,1].axis('off')

    ko = kun_old(img)
    axes[row,2].imshow(ko, cmap='hot', vmin=0, vmax=1)
    axes[row,2].set_title(f'旧坤(颜色方差) avg={ko.mean():.3f}',fontsize=9)
    axes[row,2].axis('off')

    # lap密度
    lap = np.abs(laplace(img[:,:,0]))
    ld = bf(lap, 31, mode='reflect')
    axes[row,3].imshow(ld, cmap='magma')
    axes[row,3].set_title(f'Laplacian密度 avg={ld.mean():.4f}',fontsize=9)
    axes[row,3].axis('off')

plt.tight_layout()
out='test_output/test_kun_grass.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

for name, img in imgs.items():
    kn = kun_new(img)
    ko = kun_old(img)
    print(f'{name}: 新坤(结构)={kn.mean():.3f}  旧坤(颜色)={ko.mean():.3f}  ' +
          ('✓ 新>旧 草地被识别为平坦' if kn.mean()>ko.mean() else ''))
