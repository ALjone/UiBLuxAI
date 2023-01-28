from typing import List, Tuple
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical


class block(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size):
        
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
        
        self.blocks.append(block(23, 64, 4))

        for _ in range(9):
            self.blocks.append(block(64, 64))

        self.linear = nn.Linear(100, 7)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()

        for layer in self.blocks:
            x = layer(x)
        
        x = self.linear(x.flatten(1))

        return F.softmax(x, dim=3).squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)