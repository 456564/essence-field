"""
① 基础视觉层 BaseVision

纯特征提取，无任何识别机制。
不知道自己在看什么，只做像素→特征图的变换。

类比：V1 皮层 — 响应边缘/朝向/纹理，但不"知道"那是物体。
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class BaseVision(nn.Module):
    """
    像素 → 层次化特征图

    输入:  [B, 3, 224, 224]  图像
    输出:  [B, 768, 7, 7]    特征图

    无分类头、无语义、无识别。
    可被 ② 预测层的反馈信号调制。
    """

    def __init__(self, pretrained=True, trainable=True):
        super().__init__()

        if pretrained:
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            model = convnext_tiny(weights=None)

        # 只取特征部分，彻底去掉分类头
        self.features = model.features

        if not trainable:
            self._freeze()

    def _freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, feedback=None):
        """
        x:         [B, 3, 224, 224]  图像
        feedback:  [B, 768, 7, 7]   来自 ② 的期望特征（调制信号）

        反馈机制：预测层告诉视觉层"你应该往这个方向提取"
        """
        # feedback 注入到每个 stage（未来可做逐层反馈）
        # 当前简化：只在第一层之前做整体调制
        if feedback is not None:
            # 反馈调制：让初始特征偏向预测方向
            x = x + feedback.mean(dim=[2, 3], keepdim=True) * 0.01

        return self.features(x)
