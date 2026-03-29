from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class ModalityAwareSelectiveFusion(BaseModule):

    def __init__(self,
                 channels: int,
                 hidden_channels: Optional[int] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = int(channels)
        if hidden_channels is None:
            self.gate = nn.Conv2d(self.channels * 2, self.channels, 1)
        else:
            hidden_channels = int(hidden_channels)
            self.gate = nn.Sequential(
                nn.Conv2d(self.channels * 2, hidden_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, self.channels, 1),
            )

    def forward(self, f_rgb: Tensor, f_depth: Tensor) -> Tensor:
        if f_rgb.shape != f_depth.shape:
            raise ValueError(
                f'Expected same shape, got rgb={tuple(f_rgb.shape)} depth={tuple(f_depth.shape)}'
            )
        if f_rgb.size(1) != self.channels:
            raise ValueError(
                f'Expected channels={self.channels}, got {int(f_rgb.size(1))}'
            )
        w_d = self.gate(torch.cat([f_rgb, f_depth], dim=1)).sigmoid()
        return f_rgb + w_d * f_depth


MSF = ModalityAwareSelectiveFusion
