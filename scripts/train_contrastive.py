"""
对比学习训练 v2 — 固定算子, 学投影+融合

改进:
- 简单颜色增强 (RGB空间, 不用HSV)
- 投影头 MLP (27→64→27) 增加非线性
- 正交随机初始化 (打破恒等)
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, random, argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.pipeline import PhysicalPipeline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path(__file__).resolve().parent.parent / 'checkpoints'
OUT.mkdir(exist_ok=True)


# ═══════════════════════════════════════
# Simple RGB augmentation (no HSV)
# ═══════════════════════════════════════

def augment(img_hwc):
    h, w = img_hwc.shape[:2]
    img = img_hwc.copy()

    # Brightness
    b = 0.8 + random.random() * 0.4
    img = img * b

    # Contrast (around mean)
    if random.random() < 0.8:
        mean = img.mean(axis=(0,1), keepdims=True)
        c = 0.5 + random.random() * 1.0
        img = (img - mean) * c + mean

    # Saturation (blend with grayscale)
    if random.random() < 0.8:
        gray = img.mean(axis=2, keepdims=True)
        s = random.random() * 0.8
        img = img * (1 - s) + gray * s

    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.randn(h, w, 3).astype(np.float32) * 0.02
        img = img + noise

    # Horizontal flip
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()

    # Random crop + resize
    scale = 0.85 + random.random() * 0.15
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h < h and new_w < w:
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        img = img[top:top+new_h, left:left+new_w]
    img = cv2.resize(img, (w, h))

    return np.clip(img, 0, 1).astype(np.float32)


# ═══════════════════════════════════════
# Dataset
# ═══════════════════════════════════════

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.files.extend(list(Path(root_dir).rglob(ext)))
        if not self.files:
            raise FileNotFoundError(f"No images in {root_dir}")
        self.img_size = img_size
        print(f"Found {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx % len(self.files)]
        img = cv2.imread(str(fp))
        if img is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = img.shape[:2]
        s = self.img_size
        # Center crop to square
        if h > w:
            top = (h - w) // 2; img = img[top:top+w]
        else:
            left = (w - h) // 2; img = img[:, left:left+h]
        img = cv2.resize(img, (s, s))
        return torch.from_numpy(img).permute(2, 0, 1)


# ═══════════════════════════════════════
# Projection head (small MLP)
# ═══════════════════════════════════════

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


# ═══════════════════════════════════════
# Training
# ═══════════════════════════════════════

def train(args):
    dataset = ImageFolderDataset(args.data, img_size=args.size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)

    pipe = PhysicalPipeline().to(DEVICE)
    proj_head = ProjectionHead(in_dim=27, hidden=128, out_dim=64).to(DEVICE)

    # Freeze operators
    for name, p in pipe.named_parameters():
        p.requires_grad = ('operator_layer' not in name)

    n_trainable = sum(p.numel() for p in pipe.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in proj_head.parameters())
    print(f"Trainable: pipe={n_trainable} + head={n_proj} = {n_trainable+n_proj}")
    print(f"Device: {DEVICE}, Size: {args.size}, Batch: {args.batch}")
    print(f"Temperature: {args.tau}, LR: {args.lr}")

    optimizer = torch.optim.AdamW(
        list(pipe.parameters()) + list(proj_head.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    for epoch in range(args.epochs):
        pipe.train(); proj_head.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for imgs in pbar:
            B = imgs.shape[0]
            # Two augmented views
            imgs_np = imgs.permute(0,2,3,1).cpu().numpy()
            v1 = torch.stack([torch.from_numpy(augment(imgs_np[i])).permute(2,0,1) for i in range(B)]).to(DEVICE)
            v2 = torch.stack([torch.from_numpy(augment(imgs_np[i])).permute(2,0,1) for i in range(B)]).to(DEVICE)

            # Forward pipeline
            f1 = pipe(v1)  # [B, 27, H, W]
            f2 = pipe(v2)

            # Global avg pool
            z1 = f1.mean(dim=(2, 3))  # [B, 27]
            z2 = f2.mean(dim=(2, 3))

            # Projection head
            z1 = proj_head(z1)  # [B, 64]
            z2 = proj_head(z2)

            # InfoNCE
            z = torch.cat([z1, z2], dim=0)  # [2B, 64]
            sim = torch.mm(z, z.T) / args.tau  # [2B, 2B]

            # Mask self
            mask = torch.eye(2*B, device=DEVICE).bool()
            sim = sim.masked_fill(mask, -1e9)

            # z1[i] positive = z2[i] = index B+i
            pos_idx = torch.arange(B, 2*B, device=DEVICE)

            # z1 as anchor
            logits_1 = sim[:B]  # [B, 2B]
            labels_1 = pos_idx

            # z2 as anchor
            logits_2 = sim[B:]  # [B, 2B]
            labels_2 = torch.arange(B, device=DEVICE)

            loss = (F.cross_entropy(logits_1, labels_1) + F.cross_entropy(logits_2, labels_2)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proj_head.parameters(), 1.0)
            optimizer.step()
            pipe.clamp_weights()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if (epoch+1) % args.save_every == 0:
            ckpt = OUT / f'contrastive_epoch{epoch+1}.pth'
            torch.save({'epoch': epoch+1, 'pipe': pipe.state_dict(),
                        'head': proj_head.state_dict(), 'loss': avg_loss}, ckpt)
            print(f"  Saved: {ckpt}")

    ckpt = OUT / 'contrastive_final.pth'
    torch.save({'pipe': pipe.state_dict(), 'head': proj_head.state_dict()}, ckpt)
    print(f"Done: {ckpt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'data' / 'caltech101' / '101_ObjectCategories'))
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--save_every', type=int, default=5)
    args = parser.parse_args()
    train(args)
