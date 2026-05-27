"""
八卦→64卦 核心模型

8 卦子网络 → 外积 → 64 维卦象 → 分类/解码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .subnets import BAGUA_REGISTRY


class HexagramModel(nn.Module):
    """
    八卦→64卦 视觉模型

    8 子网络按万物类象提取特征 → 外积生成 64 维卦象 → 分类头
    """
    def __init__(self, num_classes=5):
        super().__init__()

        self.subnets = nn.ModuleDict({
            name: constr() for name, _, _, constr in BAGUA_REGISTRY
        })

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, return_all=False):
        # 8 卦子网络并行提取特征
        features = {}
        for name, net in self.subnets.items():
            features[name] = net(x)  # [B, 8]

        # 外积 → 64 卦
        stacked = torch.stack(list(features.values()), dim=1)  # [B, 8, 8]
        B = stacked.size(0)
        hexagram = torch.bmm(stacked, stacked.transpose(1, 2))  # [B, 8, 8]
        hexagram = hexagram.view(B, -1)  # [B, 64]

        logits = self.classifier(hexagram)

        if return_all:
            return logits, hexagram, features
        return logits


class HexagramWithConfidence(nn.Module):
    """
    带置信度的八卦→64卦模型（TODO）

    每个子网络输出：
      特征向量 (8维) + 置信度 (0~1) + 空间响应图 (H×W)
    """
    pass
