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
from tqdm import tqdm

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

    # 冻结所有参数，仅解冻 W_up/W_dn、投影层
    for param in model.parameters():
        param.requires_grad = False
    model.fusion.W_up.requires_grad = True
    model.fusion.W_dn.requires_grad = True
    for param in model.operator_layer.projections.parameters():
        param.requires_grad = True
    
    # 颜色调制版的 1×1 颜色卷积
    if hasattr(model.operator_layer.base_ops, 'parameters'):
        for param in model.operator_layer.base_ops.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': [model.fusion.W_up, model.fusion.W_dn], 'lr': args.lr_A},
        {'params': model.operator_layer.projections.parameters(), 'lr': args.lr_proj},
    ])
    # 如果存在颜色卷积，加进去（用稍高的学习率）
    if hasattr(model.operator_layer.base_ops, 'parameters'):
        color_params = list(model.operator_layer.base_ops.parameters())
        if color_params:
            optimizer.add_param_group({'params': color_params, 'lr': args.lr_A * 3})

    print(f"可训练参数:")
    print(f"  W_up/dn: {model.fusion.W_up.numel()+model.fusion.W_dn.numel()} 个参数")
    proj_params = sum(p.numel() for p in model.operator_layer.projections.parameters())
    print(f"  投影层: {proj_params} 个参数")
    if hasattr(model.operator_layer.base_ops, 'parameters'):
        color_params_count = sum(p.numel() for p in model.operator_layer.base_ops.parameters())
        print(f"  颜色卷积层: {color_params_count} 个参数")

    # 早停
    best_loss = float('inf')
    patience_counter = 0
    patience = args.patience

    # 学习率余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * min(args.max_batches or len(loader), len(loader)),
        eta_min=args.lr_A * 0.01)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_valid = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    total=args.max_batches or len(loader), leave=False)
        for batch_idx, (img, _) in enumerate(pbar):
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
            for proj in model.operator_layer.projections.values():
                proj.weight.data.clamp_(min=0)
            model.fusion.W_up.data.clamp_(min=0)
            model.fusion.W_dn.data.clamp_(min=0)
            scheduler.step()

            total_loss += loss.item()
            n_valid += 1
            pbar.set_postfix(loss=f"{total_loss/n_valid:.6f}")

        pbar.close()
        avg_loss = total_loss / max(n_valid, 1)
        tqdm.write(f"=== Epoch {epoch+1}/{args.epochs} avg loss {avg_loss:.6f} ===")

        # 早停
        if avg_loss < best_loss - args.min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"早停: {patience}轮未改善，停止训练")
                break

        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_dict = {
                'W_up': model.fusion.W_up.data,
                'W_dn': model.fusion.W_dn.data,
                'proj': model.operator_layer.projections.state_dict(),
                'epoch': epoch + 1,
            }
            path = os.path.join(args.save_dir, f'bootstrap_epoch{epoch+1}.pth')
            torch.save(save_dict, path)
            print(f"  已保存: {path}")

            # 生成最强卦复合图（取第一批数据的第一张）
            from src.visualize import argmax_gua_composite
            img_np = (img[0].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            comp = argmax_gua_composite(field.detach(), img_np)
            comp_path = os.path.join(args.save_dir, f'epoch{epoch+1:02d}_composite.png')
            import cv2
            cv2.imwrite(comp_path, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            tqdm.write(f"  复合图: {comp_path}")


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
    parser.add_argument('--max_batches', type=int, default=0,
                        help='每轮最多处理多少张（0=全部）')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停：连续几轮无改善则停止')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='早停：最小改善阈值')
    args = parser.parse_args()
    train(args)
