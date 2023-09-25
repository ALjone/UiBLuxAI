from typing import Tuple
import torch
from torch import nn
from .blocks import ConvBlock


class actor(nn.Module):
    def __init__(self, intput_channels, unit_action_space:int, n_blocks:int, intermediate_channels:int, use_batch_norm = False) -> None:
        super(actor, self).__init__()
        
        layer = ConvBlock

        blocks = []

        #Make shared part
        blocks.append(nn.Conv2d(intput_channels, intermediate_channels, kernel_size=5, padding = 2))
        blocks.append(nn.LeakyReLU())
        for _ in range(n_blocks):
            blocks.append(layer(intermediate_channels, intermediate_channels, kernel_size=5))
            if use_batch_norm:
                blocks.append(nn.BatchNorm2d(intermediate_channels))


        #Make global features part

        self.conv = nn.Sequential(*blocks)

        self.unit_output = nn.Conv2d(intermediate_channels, unit_action_space, 1)
        
        self.factory_output = nn.Conv2d(intermediate_channels, 4, 1)

    def forward(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(image_features.shape) == 3:
            image_features = image_features.unsqueeze(0)

        x = self.conv(image_features)

        x_unit = self.unit_output(x)
        x_factory = self.factory_output(x)

        return x_unit.permute(0, 2, 3, 1), x_factory.permute(0, 2, 3, 1)
        

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)