"""
八卦→64卦 测试入口

用法:
  python scripts/test_bagua.py --image test_maccup.png
  python scripts/test_bagua.py --image test_maccup.png --viz
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import HexagramModel
from src.subnets import BAGUA_REGISTRY
from src.viz import visualize_all, compute_activation_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--viz", action="store_true", help="显示空间激活图")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载模型
    model = HexagramModel(num_classes=5).to(device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total/1e3:.0f}K")

    # 加载图片
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 前向
    with torch.no_grad():
        logits, hexagram, features = model(img_tensor, return_all=True)

    hex_64 = hexagram[0].cpu().numpy()
    hex_mat = hex_64.reshape(8, 8)

    names_cn = [cn for _, cn, _, _ in BAGUA_REGISTRY]

    # ─── 八卦特征 ───
    print(f"\n{'='*60}")
    print("八卦特征向量")
    print(f"{'='*60}")
    for name, cn, desc, _ in BAGUA_REGISTRY:
        vals = features[name][0].cpu().numpy()
        mag = np.linalg.norm(vals)
        bar = '█' * int(mag * 8)
        print(f"  {cn}: |{mag:.3f}| {bar}  [{', '.join(f'{v:+.2f}' for v in vals[:4])}...]")

    # ─── 64卦分析 ───
    pairs = []
    for i in range(8):
        for j in range(i, 8):
            pairs.append((abs(hex_mat[i, j]), i, j))
    pairs.sort(reverse=True)

    print(f"\n{'='*60}")
    print("64卦分析（最强的 6 对卦象关系）")
    print(f"{'='*60}")
    for val, i, j in pairs[:6]:
        print(f"  {names_cn[i]:4s} × {names_cn[j]:4s} = {val:+.4f}")

    # ─── 可视化 ───
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    if args.viz:
        img_disp = cv2.resize(img_rgb, (224, 224))
        out_path = str(output_dir / f"bagua_activation_{Path(args.image).stem}.png")
        visualize_all(img_tensor, img_disp, out_path)
        print(f"\n激活图已保存: {out_path}")


if __name__ == "__main__":
    main()
