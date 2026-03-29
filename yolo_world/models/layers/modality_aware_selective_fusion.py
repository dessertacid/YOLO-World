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
                 in_modalities: int = 2,
                 hidden_channels: Optional[int] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = int(channels)
        if in_modalities not in {2, 3}:
            raise ValueError(f'Unsupported in_modalities={in_modalities}')
        self.in_modalities = int(in_modalities)
        if hidden_channels is None:
            self.gate = nn.Conv2d(self.channels * self.in_modalities,
                                  self.channels, 1)
        else:
            hidden_channels = int(hidden_channels)
            self.gate = nn.Sequential(
                nn.Conv2d(self.channels * self.in_modalities, hidden_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, self.channels, 1),
            )

    def forward(self,
                f_rgb: Tensor,
                f_depth: Tensor,
                f_prev: Optional[Tensor] = None) -> Tensor:
        if f_rgb.shape != f_depth.shape:
            raise ValueError(
                f'Expected same shape, got rgb={tuple(f_rgb.shape)} depth={tuple(f_depth.shape)}'
            )
        if f_rgb.size(1) != self.channels:
            raise ValueError(
                f'Expected channels={self.channels}, got {int(f_rgb.size(1))}'
            )
        if self.in_modalities == 3:
            if f_prev is None:
                raise ValueError('f_prev is required when in_modalities=3')
            if f_prev.shape != f_rgb.shape:
                raise ValueError(
                    f'Expected same shape for f_prev, got prev={tuple(f_prev.shape)} rgb={tuple(f_rgb.shape)}'
                )
            gate_in = torch.cat([f_rgb, f_depth, f_prev], dim=1)
        else:
            gate_in = torch.cat([f_rgb, f_depth], dim=1)
        w_d = self.gate(gate_in).sigmoid()
        return f_rgb + w_d * f_depth


MSF = ModalityAwareSelectiveFusion
