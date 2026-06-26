"""
物质读出头

从本质场中读取物质信息：
  输入: 64 维场 [B, 64, H, W]
  输出: 物质列表 [(掩码, 基元向量, 元信息), ...]

支持:
  - 自动分割（扩散 + 聚类）
  - 物质属性分析（算子贡献解释）
  - 跨图物质匹配
"""

import torch
import numpy as np
from .diffusion import FieldDiffusion
from .primitive import discover_primitives, primitive_to_operator_profile


class MaterialReader:
    """
    物质读取器。

    流程: 本质场 → 扩散 → 基元发现 → 物质信息

    用法:
        reader = MaterialReader(n_clusters=None)  # 自动选 K
        materials = reader.read(field_64, operator_maps)
        for mat in materials:
            print(mat['area'], mat['operator_profile'])
    """

    def __init__(self, n_clusters=None, cluster_method='kmeans',
                 tau=0.3, alpha=0.5, n_iters=10):
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self.diffusion = FieldDiffusion(
            tau=tau, alpha=alpha, n_iters=n_iters, kernel_size=5
        )

    def read(self, field_64, operator_maps, device=None):
        """
        从本质场读取物质。

        Args:
            field_64: [B, 64, H, W] 本质场
            operator_maps: Dict[name -> [B, 1, H, W]] 原始算子

        Returns:
            materials: list[dict] 按面积降序排列
                每项 = {
                    'mask': [H, W] bool,
                    'prototype': [64] ndarray,
                    'area_ratio': float,
                    'operator_profile': [8] ndarray,
                    'label': int,
                }
            field_diffused: [B, 64, H, W]
            scores: dict 聚类质量
        """
        if device is not None:
            self.diffusion = self.diffusion.to(device)

        B, C, H, W = field_64.shape

        # 1. 场自组织扩散
        field_diff, convergence = self.diffusion(field_64, operator_maps)

        # 2. 基元发现
        labels, prototypes, scores = discover_primitives(
            field_diff.detach().cpu(),
            method=self.cluster_method,
            n_clusters=self.n_clusters,
        )

        # 3. 构建物质列表
        materials = []
        for k in range(len(prototypes)):
            mask = (labels == k)
            area = mask.sum() / (H * W)
            profile = primitive_to_operator_profile(prototypes[k])
            materials.append({
                'mask': mask,
                'prototype': prototypes[k],
                'area_ratio': float(area),
                'operator_profile': profile,
                'label': k,
            })

        # 按面积降序
        materials.sort(key=lambda m: m['area_ratio'], reverse=True)

        scores['convergence'] = convergence
        return materials, field_diff, scores

    def describe(self, materials, operator_names=None):
        """
        生成物质的人类可读描述。

        Args:
            materials: read() 的输出
            operator_names: list[str] 算子名称

        Returns:
            description: str
        """
        if operator_names is None:
            operator_names = [
                'qian(圆度)', 'kun(容器)', 'zhen(边缘)', 'xun(纹理)',
                'kan(曲率)', 'li(能量)', 'gen(块状)', 'dui(对比)'
            ]

        lines = []
        lines.append(f"=== 物质分析 ({len(materials)} 种物质) ===")
        for i, mat in enumerate(materials):
            profile = mat['operator_profile']
            top_ops = np.argsort(-profile)[:3]
            ops_desc = [
                f"{operator_names[j]}={profile[j]:.3f}"
                for j in top_ops
            ]
            lines.append(
                f"物质{i+1}: 面积={mat['area_ratio']*100:.1f}%  "
                f"主算子: {', '.join(ops_desc)}"
            )
        return '\n'.join(lines)


def test_reader():
    """自测"""
    import sys
    sys.path.insert(0, '.')
    from src.pipeline import BaguaPipeline
    import cv2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = BaguaPipeline(d=8).to(device).eval()
    ckpt = torch.load('checkpoints_mini/bootstrap_epoch20.pth',
                      map_location=device)
    pipe.fusion.A.data = ckpt['A']
    pipe.operator_layer.projections.load_state_dict(ckpt['proj'])

    img = cv2.imread('test_maccup.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    x = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    x = x.unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        field_64, ops = pipe(x, return_operators=True)

    reader = MaterialReader(n_clusters=None)
    reader.diffusion = reader.diffusion.to(device)
    materials, field_diff, scores = reader.read(field_64, ops, device)

    print(f"聚类数: {scores['n_clusters']}")
    print(f"Davies-Bouldin: {scores['davies_bouldin']:.3f}")
    print(f"收敛: {scores['convergence'][-1]:.6f}")
    print(reader.describe(materials))
    return materials, field_diff, scores


if __name__ == '__main__':
    test_reader()
