from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class LearnableDepthCalibration(BaseModule):

    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: Optional[int] = None,
                 init_scale: float = 1.0,
                 init_bias: float = 0.0,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        if hidden_channels is None:
            hidden_channels = in_channels

        self.in_channels = int(in_channels)
        self.scale = nn.Parameter(torch.full((1, self.in_channels, 1, 1),
                                             float(init_scale)))
        self.bias = nn.Parameter(torch.full((1, self.in_channels, 1, 1),
                                            float(init_bias)))

        self.denoise = nn.Sequential(
            nn.Conv2d(self.in_channels, int(hidden_channels), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_channels), self.in_channels, 3, padding=1),
        )

    def forward(self, depth: Tensor) -> Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() != 4:
            raise ValueError(f'Expected depth with 3 or 4 dims, got {depth.dim()}')
        if depth.size(1) != self.in_channels:
            raise ValueError(
                f'Expected depth channels={self.in_channels}, got {depth.size(1)}'
            )

        depth_calib = depth * self.scale + self.bias
        residual = self.denoise(depth_calib)
        return depth_calib + residual


LDC = LearnableDepthCalibration
