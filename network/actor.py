from typing import List, Tuple
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical
from .blocks import ResConvBlock, ResSEBlock


class actor(nn.Module):
    def __init__(self, intput_channels, unit_action_space:int = 7, factory_action_space:int = 3,  n_blocks:int = 5, n_blocks_factories_units:int = 2,
                  intermediate_channels:int = 64, layer_type = "SE") -> None:
        super(actor, self).__init__()
        
        if layer_type == "SE":
            layer = ResSEBlock
        elif layer_type == "conv":
            layer = ResConvBlock
        else:
            raise ValueError(f"{layer_type} is not a valid layer type")
        blocks = []
        blocks_factory = []
        blocks_units = []

        #Make shared part
        blocks.append(nn.Conv2d(intput_channels, intermediate_channels, kernel_size=3, padding = 1))
        for _ in range(n_blocks-2):
            blocks.append(layer(intermediate_channels, intermediate_channels))
        blocks.append(layer(intermediate_channels, intermediate_channels))

        #Make robot part
        for _ in range(n_blocks_factories_units):
            blocks_units.append(layer(intermediate_channels, intermediate_channels))
        blocks_units.append(nn.Conv2d(intermediate_channels, unit_action_space, 1))

        #Make factory part
        for _ in range(n_blocks_factories_units):
            blocks_factory.append(layer(intermediate_channels, intermediate_channels))
        blocks_factory.append(nn.Conv2d(intermediate_channels, factory_action_space, 1))

        self.shared_conv = nn.Sequential(*blocks)
        self.unit_conv = nn.Sequential(*blocks_units)
        self.factory_conv = nn.Sequential(*blocks_factory)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()

        x = self.shared_conv(x)
        x_robot = self.unit_conv(x)
        x_factory = self.factory_conv(x)

        x_robot = x_robot.permute(0, 2, 3, 1)
        x_factory = x_factory.permute(0, 2, 3, 1)

        return F.softmax(x_robot, dim=3).squeeze(), F.softmax(x_factory, dim=3).squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)