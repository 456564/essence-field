"""
三层闭环模型最终版

分工：
  ① BaseVision      纯特征提取，无识别
  ② WorldPredictor  物理世界知识，预测+比较+反馈
  ③ Classifier      最终决策层，轻量分类器

循环：
  特征 → 感知物理量 → 状态 → 预测预期 → 
    误差 → 反馈 → 重新提取 → 感知物理量 → ... → 收敛 → 分类
"""

import torch
import torch.nn as nn

from .backbone import BaseVision
from .world_predictor import WorldPredictor


class ClosedLoopVision(nn.Module):
    """
    类人视觉闭环模型

    ① 不知道看到什么（纯特征提取）
    ② 知道物理世界规律（预测层 = 智能层）
    ③ 做最终决定（轻量分类器）
    """

    def __init__(self, num_iters=2, state_dim=256,
                 pretrained_backbone=True, trainable_backbone=True):
        super().__init__()

        # ① 视觉层 — 无识别，纯特征
        self.backbone = BaseVision(
            pretrained=pretrained_backbone,
            trainable=trainable_backbone,
        )

        # ② 预测层 — 物理世界知识在此
        self.predictor = WorldPredictor(
            feat_dim=768,
            state_dim=state_dim,
        )

        # ③ 决策层 — 轻量分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 1000),
        )

        self.num_iters = num_iters

    def forward(self, x, return_all=False):
        """
        一次 forward = 多轮闭环迭代

        x: [B, 3, 224, 224]
        """
        B = x.size(0)
        device = x.device

        states = []
        errors = []
        feedback = None

        for i in range(self.num_iters):
            # ① 视觉特征提取（可受反馈调制）
            features = self.backbone(x, feedback=feedback)

            # ② 物理世界预测
            out = self.predictor(features)

            state = out["state"]
            error_norm = out["error_norm"]

            states.append(state)
            errors.append(error_norm)

            # 如果不是最后一轮：生成反馈重新看
            if i < self.num_iters - 1:
                feedback = out["feedback"]
            else:
                feedback = None

            # 误差收敛提前退出
            if error_norm.mean().item() < 0.5 and i > 0:
                break

        # ③ 决策层用最终状态做分类
        logits = self.classifier(states[-1])

        if return_all:
            return {
                "logits": logits,
                "state_final": states[-1],
                "states": states,
                "errors": errors,
            }

        return logits
