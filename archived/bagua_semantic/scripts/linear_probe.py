"""
线性分类探测：验证64维场包含语义信息

固定算子+投影层+A核，提取每张图的64维场 → 全局平均池化 → 64维向量
训练线性分类器（64→101类），看准确率是否远超随机（~1%）。
"""

import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from src.pipeline import BaguaPipeline


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    DATA = "data/caltech101/101_ObjectCategories"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=DATA, transform=transform)

    # 排除 BACKGROUND_Google（102类，多了一个背景类）
    classes = [c for c in dataset.classes if c != "BACKGROUND_Google"]
    keep_idx = [i for i, (_, lbl) in enumerate(dataset.samples)
                if dataset.classes[lbl] != "BACKGROUND_Google"]
    dataset = Subset(dataset, keep_idx)
    dataset.classes = classes
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"数据集: {len(dataset)} 张, {len(classes)} 类")

    # ─── 加载训练后的模型 ───
    model = BaguaPipeline().to(device).eval()
    import glob
    ckpt_path = sorted(glob.glob("checkpoints_fixedcolor/bootstrap_epoch*.pth"))[-1]
    print(f"加载: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.fusion.W_up.data = ckpt.get("W_up", ckpt.get("A")); model.fusion.W_dn.data = ckpt.get("W_dn", ckpt.get("A"))
    model.operator_layer.projections.load_state_dict(ckpt['proj'])
    print("加载了训练后的权重")

    # ─── 提取 64 维场特征（全局平均池化） ───
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for img, label in tqdm(loader, desc="提取特征"):
            img = img.to(device)
            field = model(img)                    # [1, 64, H, W]
            feat = field.mean(dim=[2, 3])         # [1, 64]
            all_feats.append(feat.cpu().numpy())
            all_labels.append(label.item())

    X = np.concatenate(all_feats, axis=0)          # [N, 64]
    y = np.array(all_labels)
    print(f"特征矩阵: {X.shape}")

    del model  # 释放显存

    # ─── 线性分类 ───
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=5000, C=1.0, n_jobs=4)
    clf.fit(X_scaled, y)
    train_acc = clf.score(X_scaled, y)
    print(f"\n逻辑回归训练集准确率: {train_acc*100:.1f}%")

    cv_scores = cross_val_score(clf, X_scaled, y, cv=3)
    print(f"3折交叉验证: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # ─── 随机权重基线 ───
    model_rnd = BaguaPipeline().to(device).eval()
    all_feats_rnd = []
    with torch.no_grad():
        for img, label in tqdm(loader, desc="随机权重特征"):
            img = img.to(device)
            field = model_rnd(img)
            feat = field.mean(dim=[2, 3])
            all_feats_rnd.append(feat.cpu().numpy())
    X_rnd = np.concatenate(all_feats_rnd, axis=0)
    X_rnd_scaled = scaler.fit_transform(X_rnd)
    clf_rnd = LogisticRegression(max_iter=5000, C=1.0, n_jobs=4)
    clf_rnd.fit(X_rnd_scaled, y)
    rnd_acc = clf_rnd.score(X_rnd_scaled, y)

    # ─── 结果汇总 ───
    print(f"\n{'='*55}")
    print(f"线性分类探测结果")
    print(f"{'='*55}")
    print(f"  训练后准确率:  {train_acc*100:.1f}%")
    print(f"  3折交叉验证:   {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
    print(f"  随机权重基线:  {rnd_acc*100:.1f}%")
    print(f"  Caltech101 随机猜测: ~1%")
    print(f"{'='*55}")

    if train_acc > 0.30:
        print("\n✅ 通过: 64维场编码了语义信息 (准确率>30%)")
    elif train_acc > 0.10:
        print("\n⚠️ 部分通过: 有语义信号但较弱 (10%~30%)")
    else:
        print("\n❌ 未通过: 64维场仅有定位能力 (<10%)")


if __name__ == '__main__':
    main()
