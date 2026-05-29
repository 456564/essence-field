"""
显著性检测：64维场范数 → 显著图

在 DUTS-TE 上验证64维场能否直接预测人眼注视点。
不需要训练，直接推理。与随机场和梯度幅值对比。

下载地址：http://saliencydetection.net/duts/
"""

import sys, torch, torch.nn.functional as F, numpy as np, cv2, os, zipfile, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm
from src.pipeline import BaguaPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── 下载 DUTS-TE ───
DUTS_DIR = Path("data/DUTS-TE")
if not DUTS_DIR.exists():
    print("下载 DUTS-TE 数据集...")
    url = "http://saliencydetection.net/duts/download/DUTS-TE.zip"
    zip_path = "data/DUTS-TE.zip"
    # 下载
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc="下载") as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    # 解压
    print("解压中...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("data/")
    os.remove(zip_path)
    print("解压完成")

# DUTS-TE 结构：DUTS-TE/DUTS-TE-Image/ 和 DUTS-TE/DUTS-TE-Mask/
img_dir = DUTS_DIR / "DUTS-TE-Image"
gt_dir = DUTS_DIR / "DUTS-TE-Mask"
img_files = sorted(img_dir.glob("*.jpg"))
print(f"DUTS-TE: {len(img_files)} 张图片")


# ─── 加载模型 ───
model = BaguaPipeline().to(device).eval()
import glob
ckpt_path = sorted(glob.glob("checkpoints_fixedcolor/bootstrap_epoch*.pth"))[-1]
print(f"加载: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
model.fusion.W_up.data = ckpt.get("W_up", ckpt.get("A")); model.fusion.W_dn.data = ckpt.get("W_dn", ckpt.get("A"))
model.operator_layer.projections.load_state_dict(ckpt['proj'])
print("加载了训练后的权重")


def norm_saliency(model, img_tensor):
    """64维场范数 → 显著图"""
    with torch.no_grad():
        field = model(img_tensor)          # [1, 64, H, W]
        sal = field[0].norm(dim=0)         # [H, W]
    # 归一化到 [0,1]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.cpu().numpy()


def gradient_saliency(img_tensor):
    """RGB 梯度幅值作为基线"""
    sobel = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=img_tensor.dtype, device=device)
    gx = torch.cat([F.conv2d(img_tensor[:,c:c+1,:,:], sobel, padding=1) for c in range(3)], dim=1)
    gy = torch.cat([F.conv2d(img_tensor[:,c:c+1,:,:], sobel.transpose(2,3), padding=1) for c in range(3)], dim=1)
    sal = torch.sqrt((gx**2 + gy**2).sum(dim=1, keepdim=True))  # [1,1,H,W]
    sal = sal[0,0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.cpu().numpy()


def compute_metrics(sal_map, gt_map):
    """MAE + AUC-Judd + CC + SIM（无需二值化）"""
    sal = sal_map.astype(np.float32).ravel()
    gt = gt_map.astype(np.float32).ravel()
    n = len(sal)

    # MAE
    mae = np.abs(sal - gt).mean()

    # AUC-Judd：100 个等分阈值，TPR vs FPR
    thresholds = np.linspace(0, 1, 100)
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        pred = (sal >= t).astype(np.float32)
        tp = (pred * gt).sum()
        fp = pred.sum() - tp
        fn = gt.sum() - tp
        tn = n - tp - fp - fn
        tpr[i] = tp / (tp + fn + 1e-8)
        fpr[i] = fp / (fp + tn + 1e-8)
    # 排序（降序）并计算AUC
    idx = np.argsort(thresholds)[::-1]
    tpr_sorted = tpr[idx]
    fpr_sorted = fpr[idx]
    auc = np.trapezoid(tpr_sorted, fpr_sorted)

    # CC（皮尔逊相关系数）
    sal_norm = (sal - sal.mean()) / (sal.std() + 1e-8)
    gt_norm = (gt - gt.mean()) / (gt.std() + 1e-8)
    cc = (sal_norm * gt_norm).mean()

    # SIM（直方图交集，0~1）
    bins = 255
    h_sal, _ = np.histogram(sal, bins=bins, range=(0,1), density=False)
    h_gt, _ = np.histogram(gt, bins=bins, range=(0,1), density=False)
    h_sal = h_sal / h_sal.sum()
    h_gt = h_gt / h_gt.sum()
    sim = np.minimum(h_sal, h_gt).sum()

    return mae, auc, cc, sim


# ─── 测试 ───
print("\n测试中...")
maes_model = []; aucs_model = []; ccs_model = []; sims_model = []
maes_grad = []; aucs_grad = []; ccs_grad = []; sims_grad = []

for img_path in tqdm(img_files, desc="显著性检测"):
    # 读取图片
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 读取真值
    gt_path = gt_dir / (img_path.stem + ".png")
    gt = cv2.imread(str(gt_path), 0).astype(np.float32) / 255.0
    # 缩放到模型输入尺寸
    SIZE = 128
    img_small = cv2.resize(img_rgb, (SIZE, SIZE))
    gt_small = cv2.resize(gt, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    # 模型显著图
    x = torch.from_numpy(img_small).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    sal_model = norm_saliency(model, x)
    sal_model = cv2.resize(sal_model, (w, h))
    m, a, c, s = compute_metrics(sal_model, gt)
    maes_model.append(m); aucs_model.append(a); ccs_model.append(c); sims_model.append(s)

    # 梯度幅值基线
    sal_grad = gradient_saliency(x)
    sal_grad = cv2.resize(sal_grad, (w, h))
    m, a, c, s = compute_metrics(sal_grad, gt)
    maes_grad.append(m); aucs_grad.append(a); ccs_grad.append(c); sims_grad.append(s)

# ─── 结果 ───
print(f"\n{'='*60}")
print(f"显著性检测结果 (DUTS-TE, {len(img_files)} 张)")
print(f"{'='*60}")
print(f"  方法             MAE    AUC↑    CC↑    SIM↑")
print(f"  {'-'*46}")
print(f"  64维场范数       {np.mean(maes_model):.4f}  {np.mean(aucs_model):.4f}  "
      f"{np.mean(ccs_model):.4f}  {np.mean(sims_model):.4f}")
print(f"  梯度幅值基线     {np.mean(maes_grad):.4f}  {np.mean(aucs_grad):.4f}  "
      f"{np.mean(ccs_grad):.4f}  {np.mean(sims_grad):.4f}")
print(f"  DUTS-TE 已知基线:")
print(f"    IT (Itti 98)   0.28    -      -      -")
print(f"    AIM (2008)     0.25    -      -      -")
print(f"{'='*60}")

if np.mean(maes_model) < 0.30:
    print(f"\n✅ MAE 通过: {np.mean(maes_model):.4f} < 0.30")
else:
    print(f"\n❌ MAE 未通过: {np.mean(maes_model):.4f} > 0.30")
print(f"📊 后续与经典显著性模型的 AUC/CC/SIM 对比可查阅文献")
