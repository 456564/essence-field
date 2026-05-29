"""震快速验证"""
import sys; sys.path.insert(0,'.')
import torch,cv2,numpy as np
from src.operators import _zhen
cup=cv2.imread('test_maccup.png')
cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(224,224))
ct=torch.from_numpy(cup[:,:,0]).float().unsqueeze(0).unsqueeze(0).cuda()
z=_zhen(ct)[0,0].cpu().numpy()
c_in=z[80:140,70:140].mean()
c_edge=z[55:70,70:150].mean()
bg=z[180:,20:60].mean()
print(f'震: avg={z.mean():.3f} max={z.max():.1f}')
print(f'  杯内={c_in:.3f}  杯边={c_edge:.3f}  背景={bg:.3f}')
print(f'  边缘>内部?  {"OK" if c_edge>c_in else "FAIL"}')
