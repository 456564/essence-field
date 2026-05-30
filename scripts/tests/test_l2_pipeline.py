"""双层抽象管线测试: L1表象场 → L2抽象场"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False

device='cuda'
from src.pipeline import BaguaPipeline

print('=== L1 only (n_layers=1) ===')
pipe1=BaguaPipeline(n_layers=1).to(device).eval()
print(f'params: {sum(p.numel() for p in pipe1.parameters())}')

print('\n=== L1+L2 (n_layers=2) ===')
pipe2=BaguaPipeline(n_layers=2).to(device).eval()
print(f'params: {sum(p.numel() for p in pipe2.parameters())}')

cup=cv2.imread('test_maccup.png'); cup=cv2.cvtColor(cup,cv2.COLOR_BGR2RGB).astype(np.float32)/255
cup=cv2.resize(cup,(128,128))
x=torch.from_numpy(cup).permute(2,0,1).unsqueeze(0).to(device)

with torch.no_grad():
    f1=pipe1(x); f2=pipe2(x)

opn=['Q乾','K坤','Z震','X巽','KA坎','L离','G艮','D兑']
print('\n=== 每卦在L1 vs L2场中的L2范数 ===')
for i in range(8):
    v1=f1[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy().mean()
    v2=f2[0,i*8:(i+1)*8].norm(dim=0).cpu().numpy().mean()
    print(f'{opn[i]}: L1={v1:.1f}  L2={v2:.1f}  change={v2-v1:+.1f}')

print('\n双层管线创建成功, L2场形状:', f2.shape)
