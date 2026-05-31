"""本质空间 — 8算子场的统一封装"""
import torch
import torch.nn as nn


CHANNEL_ORDER = ['yang', 'yin', 'dong', 'jing', 'gang', 'rou', 'ju', 'san', 'void_prob']


class EssenceSpace:
    """
    封装 [B, 8, H, W] 本质空间场。
    提供命名访问和墙掩码构建。
    """

    def __init__(self, field):
        """
        field: [B, 8, H, W] from PhysicalOperatorLayer
        """
        self.field = field
        self._wall_mask = None
        self.B, self.C, self.H, self.W = field.shape

    @classmethod
    def from_image(cls, rgb, operator_layer):
        """从RGB图像构建本质空间"""
        with torch.no_grad():
            field = operator_layer(rgb)
        return cls(field)

    def get(self, name):
        """获取单个算子场 [B, 1, H, W]"""
        idx = CHANNEL_ORDER.index(name)
        return self.field[:, idx:idx+1, :, :]

    @property
    def wall_mask(self):
        """构建墙壁掩码: gang>0.1 或 yang>0.3 形成不可穿越的墙"""
        if self._wall_mask is None:
            self._wall_mask = self._build_wall()
        return self._wall_mask

    @property
    def cavity_mask(self):
        """粒子可停留的虚空区域: void_prob > 0.3"""
        return (self.get('void_prob') > 0.3).float()

    def _build_wall(self, gang_thresh=0.1, yang_thresh=0.3, dilate_k=5):
        from .operators import _box_filter
        gang = self.get('gang')
        yang = self.get('yang')
        wall_rigid = (gang > gang_thresh).float()
        wall_solid = (yang > yang_thresh).float()
        wall = torch.clamp(wall_rigid + wall_solid, 0, 1)
        wall = _box_filter(wall, k=dilate_k)
        wall = (wall > 0.01).float()
        # Exclude cavity zone from wall — particles need to enter cavities
        cavity = self.cavity_mask
        wall = wall * (1.0 - cavity)
        return wall.detach()
