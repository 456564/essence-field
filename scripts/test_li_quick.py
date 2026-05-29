"""离 = 颜色投影直通，杯子验证"""
import sys; sys.path.insert(0,'.')
import torch, cv2, numpy as np
from src.operators import FIXED_BAGUA_COLORS

img = cv2.imread('test_maccup.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()
w = torch.tensor(FIXED_BAGUA_COLORS['li']).view(1,3,1,1).cuda()
li = (x*w).sum(1, keepdim=True).clamp(min=0)[0,0].cpu().numpy()

cup_c = li[80:140,70:140]
bg = li[180:,20:60]
shadow = li[160:190,150:185]
rim = li[55:70,70:150]

print(f'离响应: 杯身={cup_c.mean():.3f}  高光={rim.mean():.3f}  阴影={shadow.mean():.3f}  背景={bg.mean():.3f}')
print(f'杯身>背景?  {"✓" if cup_c.mean()>bg.mean() else "✗"}')
print(f'高光>杯身>阴影?  {"✓" if rim.mean()>cup_c.mean()>shadow.mean() else "✗"}')
