from typing import List, Tuple
import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical


class actor(nn.Module):
    def __init__(self, intput_channels, output_robot:int = 7,output_factory:int = 3, n_blocks:int = 10,n_blocks_robots:int = 1,n_blocks_factory:int = 1,
                  squeeze_channels:int = 64) -> None:
        super(actor, self).__init__()

        #TODO: Add split for factory and unit
        
        self.blocks = torch.nn.ParameterList()
        self.blocks_factory = torch.nn.ParameterList()
        self.blocks_robots = torch.nn.ParameterList()
        self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        for _ in range(n_blocks-2):
            self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        for _ in range(n_blocks_robots):
            self.blocks_robots.append(SqueezeExcitation(intput_channels, squeeze_channels))
        for _ in range(n_blocks_factory):
            self.n_blocks_factory.append(SqueezeExcitation(intput_channels, squeeze_channels))
            
        self.blocks_robots.append(nn.Conv2d(intput_channels, output_robot, 1))
        self.blocks_robots.append(nn.Conv2d(intput_channels, output_factory, 1))

    def forward(self, x:torch.Tensor) -> Tuple(torch.Tensor, torch.Tensor):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()


        for layer in self.blocks:
            x = layer(x)
        x_robot = x
        x_facory = x 
        for layer in self.blocks_robots:
            x_robot = layer(x_robot)
        for layer in self.blocks_robots:
            x_facory = layer(x_facory)
        return x_robot.squeeze().permute(1, 2, 0), x_facory.squeeze().permute(1, 2, 0)

    def get_action_and_value(self, x, action = None):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action.shape, probs.log_prob(action).shape, probs.entropy().shape

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)