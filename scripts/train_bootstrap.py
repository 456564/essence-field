"""
自举训练 A 核 + 投影层

方案：范数中位数 → 物体/背景掩码 → InfoNCE 对比学习
正样本：物体内部像素方向对齐
负样本：物体 vs 背景方向分离

无需聚类，纯自监督。
"""

import os, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.pipeline import BaguaPipeline


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    print(f"数据集: {len(dataset)} 张图片")

    model = BaguaPipeline().to(device)

    # 冻结所有参数，仅解冻 A 核和 1×1 conv 投影层
    for param in model.parameters():
        param.requires_grad = False
    model.fusion.A.requires_grad = True
    for param in model.operator_layer.projections.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': [model.fusion.A], 'lr': args.lr_A},
        {'params': model.operator_layer.projections.parameters(), 'lr': args.lr_proj},
    ])

    print(f"可训练参数:")
    print(f"  A 核: {model.fusion.A.numel()} 个参数")
    proj_params = sum(p.numel() for p in model.operator_layer.projections.parameters())
    print(f"  投影层: {proj_params} 个参数")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_valid = 0

        for batch_idx, (img, _) in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            img = img.to(device)                 # [1, 3, H, W]
            field = model(img)                   # [1, 64, H, W]
            B, C, H, W = field.shape
            field_flat = field.view(C, -1).t()   # [H*W, 64]

            # 1. 范数中位数 → 物体/背景掩码
            norms = torch.norm(field_flat, dim=1)
            median = norms.median()
            fg_mask = norms > median

            fg_idx = torch.where(fg_mask)[0]
            bg_idx = torch.where(~fg_mask)[0]

            if len(fg_idx) < args.min_fg or len(bg_idx) < args.min_bg:
                continue

            # 2. 归一化方向向量
            v = F.normalize(field_flat, dim=1)
            v_fg = v[fg_idx]
            v_bg = v[bg_idx]

            # 3. InfoNCE 对比损失
            n_anchors = min(args.n_anchors, len(fg_idx))
            anchor_idx = torch.randperm(len(fg_idx))[:n_anchors]
            anchors = v_fg[anchor_idx]

            sim_pos = torch.matmul(anchors, v_fg.t()) / args.temperature
            sim_neg = torch.matmul(anchors, v_bg.t()) / args.temperature

            logsumexp_pos = torch.logsumexp(sim_pos, dim=1)
            logsumexp_all = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=1), dim=1)
            loss = - (logsumexp_pos - logsumexp_all).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_valid += 1

            if (batch_idx + 1) % args.print_freq == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1} | "
                      f"Loss {total_loss/n_valid:.6f}")

        avg_loss = total_loss / max(n_valid, 1)
        print(f"=== Epoch {epoch+1}/{args.epochs} avg loss {avg_loss:.6f} ===")

        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_dict = {
                'A': model.fusion.A.data,
                'proj': model.operator_layer.projections.state_dict(),
                'epoch': epoch + 1,
            }
            path = os.path.join(args.save_dir, f'bootstrap_epoch{epoch+1}.pth')
            torch.save(save_dict, path)
            print(f"  已保存: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/caltech101/101_ObjectCategories')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_A', type=float, default=0.01)
    parser.add_argument('--lr_proj', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--min_fg', type=int, default=50)
    parser.add_argument('--min_bg', type=int, default=50)
    parser.add_argument('--n_anchors', type=int, default=16)
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--max_batches', type=int, default=500,
                        help='每轮最多处理多少张（0=全部）')
    args = parser.parse_args()
    train(args)
