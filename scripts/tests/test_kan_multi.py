"""坎多图测试：真实图中暗色/低洼/阴影"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, matplotlib, cv2, glob
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device = 'cuda'
from src.operators import _kan

DATA = 'data/caltech101/101_ObjectCategories'

def load(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return cv2.resize(img, (224,224))

pairs = [('杯子', load('test_maccup.png'))]

# 找有阴影/暗区的类别
for cat in ['BACKGROUND_Google', 'barrel', 'cannon', 'cup', 'anchor',
            'buddha', 'binocular', 'cellphone', 'camera', 'ketch']:
    files = sorted(glob.glob(f'{DATA}/{cat}/*.jpg'))
    if files:
        pairs.append((cat, load(files[len(files)//2])))

fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 2.2*len(pairs)))

for row, (name, img) in enumerate(pairs):
    ch = img[:,:,0]
    ch_t = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
    kan = _kan(ch_t)[0,0].cpu().numpy()

    axes[row,0].imshow(img.clip(0,1)); axes[row,0].set_title(name,fontsize=9)
    axes[row,0].axis('off')

    p2,p98=np.percentile(kan,2),np.percentile(kan,98)
    axes[row,1].imshow(kan, cmap='hot', vmin=p2, vmax=p98)
    axes[row,1].set_title(f'坎 max={kan.max():.1f} avg={kan.mean():.2f}',fontsize=9)
    axes[row,1].axis('off')

    # 暗度图 (1-ch)
    axes[row,2].imshow(1-img.clip(0,1), cmap='gray')
    axes[row,2].set_title('暗度(1-bright)',fontsize=9); axes[row,2].axis('off')

plt.tight_layout()
out='test_output/test_kan_multi.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'→ {out}')

print()
for name, img in pairs:
    ch_t = torch.from_numpy(img[:,:,0]).float().unsqueeze(0).unsqueeze(0).to(device)
    kan = _kan(ch_t)[0,0].cpu().numpy()
    print(f'{name:>15s}: max={kan.max():.1f}  avg={kan.mean():.3f}  >0.1占比={(kan>0.1).mean()*100:.0f}%')
