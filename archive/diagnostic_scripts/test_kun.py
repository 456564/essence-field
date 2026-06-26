"""快速测试：坤升级后的分离度"""
import sys, torch
sys.path.insert(0, '.')
from src.pipeline import BaguaPipeline
import cv2, numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BaguaPipeline().to(device).eval()
ckpt = torch.load('checkpoints_fixedcolor/bootstrap_epoch15.pth', map_location=device)
model.fusion.A.data = ckpt['A']
model.operator_layer.projections.load_state_dict(ckpt['proj'])

img = cv2.imread('data/caltech101/101_ObjectCategories/cup/image_0001.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
x = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

with torch.no_grad():
    field = model(x)
norms = field[0].norm(dim=0)
median = norms.median()
fg = norms[norms>median].mean()
bg = norms[norms<=median].mean()
sep = (fg - bg) / (bg + 1e-6)
print(f'坤升级后分离度: {sep:.2f}')
