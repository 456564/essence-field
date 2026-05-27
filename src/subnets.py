"""
八卦子网络（按万物类象设计）

每个子网络按卦的核心"象"设计，检测特定的视觉特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QianSubNet(nn.Module):
    """
    乾（天☰）— 刚健、圆、君、父、金、玉
    象：圆形、主导、刚健、权威、珍贵
    设计：大感受野，检测全局/圆形结构
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 7, padding=3, dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class KunSubNet(nn.Module):
    """
    坤（地☷）— 顺、柔、载物、均、方
    象：平坦、承载、均匀、柔顺、方形、背景
    设计：大区域平均，检测平坦/背景区域
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(7, stride=4),
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class ZhenSubNet(nn.Module):
    """
    震（雷☳）— 动、决躁、奋起、青
    象：震动、突变、爆发、上升、突然变化
    设计：高通/边缘检测器，检测突变/不连续
    """
    def __init__(self):
        super().__init__()
        self.highpass = nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)
        with torch.no_grad():
            k = torch.tensor([[[[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]]]], dtype=torch.float32)
            self.highpass.weight.data = k.repeat(3, 1, 1, 1) / 9.0
        for p in self.highpass.parameters():
            p.requires_grad = False

        self.sobel_x = nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)
        self.sobel_y = nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)
        with torch.no_grad():
            sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            self.sobel_x.weight.data = sx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            self.sobel_y.weight.data = sy.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        for p in self.sobel_x.parameters():
            p.requires_grad = False
        for p in self.sobel_y.parameters():
            p.requires_grad = False

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        hp = self.highpass(x)
        gx = self.sobel_x(x)
        gy = self.sobel_y(x)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        feat = hp + grad_mag
        return self.net(feat)


class XunSubNet(nn.Module):
    """
    巽（风☴）— 入、散、长、高、白、进退不定
    象：渗透、细长、飘散、无固定形态、细节
    设计：多尺度小核，检测细长纹理/渗透/精细结构
    """
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.net = nn.Sequential(
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        if b3.shape[-1] != b1.shape[-1]:
            b3 = F.interpolate(b3, size=b1.shape[-2:], mode='bilinear')
        multi = torch.cat([b1, b2, b3], dim=1)
        return self.net(multi)


class KanSubNet(nn.Module):
    """
    坎（水☵）— 水、隐伏、矫柔、弓轮、外柔内刚
    象：流动、弯曲、渐变、隐藏、险陷、曲折
    设计：检测渐变、曲线、流动模式、深度变化
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class LiSubNet(nn.Module):
    """
    离（火☲）— 火、日、电、中女、甲胄、戈兵
    象：明亮、附着、色彩、外刚内柔、干燥
    设计：检测颜色/光照/亮度分布
    """
    def __init__(self):
        super().__init__()
        self.color_proj = nn.Conv2d(3, 16, 1)
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        c = self.color_proj(x)
        return self.net(c)


class GenSubNet(nn.Module):
    """
    艮（山☶）— 山、止、阻隔、硬、坚多节
    象：稳定、阻挡、块状、硬质、静止
    设计：大核检测块状结构、屏障、稳定的大面积区域
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DuiSubNet(nn.Module):
    """
    兑（泽☱）— 泽、悦、口舌、毁折、附决、缺
    象：开口、凹陷、破损、表面反射、缺口
    设计：检测凹陷/缺口/开口/表面反射
    """
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        local_mean = self.avg_pool(x)
        contrast = x - local_mean
        return self.net(contrast)


# ─── 注册表 ───────────────────────────────────────────────

BAGUA_REGISTRY = [
    ("qian", "乾天", "圆/全局/主导", QianSubNet),
    ("kun",  "坤地", "平坦/背景/承载", KunSubNet),
    ("zhen", "震雷", "边缘/突变/爆发", ZhenSubNet),
    ("xun",  "巽风", "细长/渗透/细节", XunSubNet),
    ("kan",  "坎水", "渐变/弯曲/流动", KanSubNet),
    ("li",   "离火", "颜色/光照/附着", LiSubNet),
    ("gen",  "艮山", "块状/屏障/稳定", GenSubNet),
    ("dui",  "兑泽", "开口/凹陷/缺损", DuiSubNet),
]
