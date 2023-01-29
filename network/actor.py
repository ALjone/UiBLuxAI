from typing import List, Tuple
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical


class block(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size):
        super(block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size),
                                  nn.BatchNorm2d(output_channels),
                                  nn.ReLU())
        
    def forward(self, x):
        return self.conv(x)


class actor(nn.Module):
    def __init__(self, intput_channels, output_robot:int = 7,output_factory:int = 3, n_blocks:int = 10,n_blocks_robots:int = 1,n_blocks_factory:int = 1,
                  squeeze_channels:int = 64) -> None:
        super(actor, self).__init__()

        #TODO: Add split for factory and unit
        
        self.blocks = torch.nn.ParameterList()
        self.blocks_factory = torch.nn.ParameterList()
        self.blocks_robots = torch.nn.ParameterList()

        mid_channels = 16

        conv_out_channels = 8
        
        self.blocks.append(block(intput_channels, mid_channels, 4))

        for _ in range(3):
            self.blocks.append(block(mid_channels, mid_channels, 4))

        self.blocks.append(block(mid_channels, conv_out_channels, 4))

        self.linear = nn.Linear(((48-3*5)**2)*conv_out_channels, output_robot)

        print("Parameters in actor:", self.count_parameters())

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()

        for layer in self.blocks:
            x = layer(x)
        
        x = self.linear(x.flatten(1))
        return F.softmax(x, dim=1).squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)