from typing import List, Tuple
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from .blocks import ResSEBlock, ConvBlock, GlobalBlock


class actor(nn.Module):
    def __init__(self, intput_channels, unit_action_space:int, factory_action_space:int,  n_blocks:int, n_blocks_factories_units:int,
                  intermediate_channels:int, config, layer_type = "conv") -> None:
        super(actor, self).__init__()
        
        if layer_type == "SE":
            layer = ResSEBlock
        elif layer_type == "conv":
            layer = ConvBlock
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

        #Make global features part
        self.global_block =  GlobalBlock(config['map_size'])

        self.shared_conv = nn.Sequential(*blocks)
        self.unit_conv = nn.Sequential(*blocks_units)
        self.factory_conv = nn.Sequential(*blocks_factory)

    def forward(self, image_features:torch.Tensor, global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(image_features) == np.ndarray:
            image_features = torch.from_numpy(image_features)
        if len(image_features.shape) == 3:
            image_features = image_features.unsqueeze(0)

        global_features = global_features.float()
        image_features = image_features.float()

        global_image_channels = self.global_block(global_features)
        image_features = torch.concatenate((image_features, global_image_channels), dim=1)  # Assumning Batch_Size x Channels x 48 x 48

        # TODO: 12 new dimensions for image_features input
        image_features = self.shared_conv(image_features)
        x_robot = self.unit_conv(image_features)
        x_factory = self.factory_conv(image_features)

        x_robot = x_robot.permute(0, 2, 3, 1)
        x_factory = x_factory.permute(0, 2, 3, 1)
        return x_robot, x_factory

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)