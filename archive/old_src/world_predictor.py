"""
② 预测层 WorldPredictor

这是整个模型的核心。"智能"在这里。

功能：
  1. 从视觉特征解码物理世界属性（深度/遮挡/前背景）
  2. 基于训练学到的物理世界知识，生成"预期特征"
  3. 比较实际 vs 预期 → 误差
  4. 误差反馈回视觉层，重新提取

类比：V2→V4→IT，知道物理世界的规律。
不认识"猫"这个类别，但知道"有毛、有四条腿、有眼睛"的物理结构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldPredictor(nn.Module):
    """
    WorldPredictor — 物理世界模型

    输入:  [B, 768, 7, 7]   视觉特征（来自 ①）
    输出:
      actual_physics  [B, 4, 7, 7]   感知到的物理量
      expected_physics [B, 4, 7, 7]  预测的物理量（用于反馈）
      state           [B, 256]        场景理解状态
      feedback        [B, 768, 7, 7]  反馈信号（送回 ①）
    """

    def __init__(self, feat_dim=768, state_dim=256):
        super().__init__()

        # === 物理解码器：特征 → 物理量 ===
        # 这是"我看到的世界物理结构是什么"
        self.phys_encoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 4, kernel_size=1),  # 4 通道: occ+depth+fg_bg(2)
        )

        # === 世界模型编码器 ===
        # 从"感知到的物理量 + 特征"压缩为场景状态
        # 这个状态是模型对当前场景的理解
        self.state_encoder = nn.Sequential(
            nn.Conv2d(feat_dim + 4, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, state_dim),
            nn.GELU(),
        )

        # === 预期生成器：state → 预期物理量 ===
        # 基于学到的物理世界知识，"脑补"应该看到什么
        self.expected_generator = nn.Sequential(
            nn.Linear(state_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4 * 7 * 7),
        )

        # === 反馈生成器：预期物理量 → 反馈信号 ===
        self.feedback_gen = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, feat_dim, kernel_size=1),
        )

        # === 增益调节 ===
        self.gain_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def decode_physics(self, features):
        """从视觉特征解码物理量 [B, 4, 7, 7]"""
        physics = self.phys_encoder(features)
        # 约束到合理范围
        occ_edge = torch.sigmoid(physics[:, 0:1])   # [0,1] 遮挡概率
        depth = torch.tanh(physics[:, 1:2])          # [-1,1] 相对深度
        fg_bg = torch.softmax(physics[:, 2:4], dim=1) # [0,1] 前景/背景
        return torch.cat([occ_edge, depth, fg_bg], dim=1)

    def encode_state(self, features, physics):
        """特征+物理量 → 场景理解状态"""
        combined = torch.cat([features, physics], dim=1)
        return self.state_encoder(combined)

    def generate_expected(self, state):
        """从状态预测预期物理量"""
        x = self.expected_generator(state)
        expected = x.view(-1, 4, 7, 7)
        return expected

    def compute_feedback(self, expected_physics):
        """预期物理量 → 反馈调制信号"""
        return self.feedback_gen(expected_physics)

    def compute_gain(self, state):
        """自适应增益"""
        return self.gain_net(state)

    def forward(self, features):
        """
        一次完整的前向（无反馈迭代版本）

        features: [B, 768, 7, 7] 来自 ①

        返回当前感知 + 状态 + 预期
        """
        # 感知物理量
        actual = self.decode_physics(features)

        # 编码场景状态
        state = self.encode_state(features, actual)

        # 生成预期（基于物理知识的"脑补"）
        expected = self.generate_expected(state)

        # 误差
        error = F.mse_loss(actual, expected.detach(), reduction='none')
        error_norm = error.mean(dim=[1, 2, 3], keepdim=True)

        # 增益
        gain = self.compute_gain(state)

        # 反馈信号
        feedback = self.compute_feedback(expected)

        return {
            "actual": actual,
            "expected": expected,
            "state": state,
            "gain": gain,
            "error_norm": error_norm,
            "feedback": feedback,
        }
