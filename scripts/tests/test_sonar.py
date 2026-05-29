"""快速测试声纳面板"""
import sys; sys.path.insert(0, '.')
from src.visualize import *
import torch; from src.pipeline import BaguaPipeline
import cv2

device='cuda'
model=BaguaPipeline().to(device).eval()
import glob
ckpt_path=sorted(glob.glob('checkpoints_fixedcolor/bootstrap_epoch*.pth'))[-1]
print(f'加载: {ckpt_path}')
ckpt=torch.load(ckpt_path,map_location=device)
model.fusion.A.data=ckpt['A']
model.operator_layer.projections.load_state_dict(ckpt['proj'])

img=cv2.imread('test_maccup.png')
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

x=torch.from_numpy(cv2.resize(img_rgb,(128,128))).permute(2,0,1).float().unsqueeze(0).to(device)/255
with torch.no_grad():
    field=model(x)

import matplotlib.pyplot as plt
full_sonar_panel(field, img_rgb)
plt.savefig('test_output/sonar_panel.png', dpi=150, bbox_inches='tight')
plt.close()
print('done → test_output/sonar_panel.png')
