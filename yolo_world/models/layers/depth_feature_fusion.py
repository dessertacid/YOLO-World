from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class DepthFeatureFusion(BaseModule):

    def __init__(self,
                 fusion_type: str = 'add',
                 fusion_indices: Sequence[int] = (1, ),
                 depth_in_channels: int = 1,
                 feat_channels: Optional[Sequence[int]] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        if fusion_type not in {'add', 'concat'}:
            raise ValueError(f'Unsupported fusion_type={fusion_type}')
        self.fusion_type = fusion_type
        self.fusion_indices = tuple(int(x) for x in fusion_indices)
        self.depth_in_channels = int(depth_in_channels)

        self._inited = False
        self.depth_proj = nn.ModuleList()
        self.fuse_proj = nn.ModuleList()

        if feat_channels is not None:
            self.init_for_channels(feat_channels)

    def init_for_channels(self, feat_channels: Sequence[int]) -> None:
        self.depth_proj = nn.ModuleList([
            nn.Conv2d(self.depth_in_channels, int(c), 1) for c in feat_channels
        ])
        if self.fusion_type == 'concat':
            self.fuse_proj = nn.ModuleList([
                nn.Conv2d(int(c) * 2, int(c), 1) for c in feat_channels
            ])
        else:
            self.fuse_proj = nn.ModuleList()

        self._inited = True

    def forward(self, rgb_feats: Tuple[Tensor, ...], depth: Tensor) -> Tuple[Tensor, ...]:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() != 4:
            raise ValueError(f'Expected depth with 3 or 4 dims, got {depth.dim()}')
        if depth.size(1) != self.depth_in_channels:
            raise ValueError(
                f'Expected depth channels={self.depth_in_channels}, got {depth.size(1)}'
            )

        if not self._inited:
            raise RuntimeError(
                'DepthFeatureFusion is not initialized. '
                'Please call init_for_channels(feat_channels) before training.'
            )

        fused_feats = []
        for i, feat in enumerate(rgb_feats):
            if i not in self.fusion_indices:
                fused_feats.append(feat)
                continue

            d = depth
            if not torch.is_floating_point(d):
                d = d.float()
            d = F.interpolate(d, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            d = self.depth_proj[i](d)

            if self.fusion_type == 'add':
                fused_feats.append(feat + d)
            else:
                fused_feats.append(self.fuse_proj[i](torch.cat([feat, d], dim=1)))

        return tuple(fused_feats)
