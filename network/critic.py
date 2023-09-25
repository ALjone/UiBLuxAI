import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical
from .blocks import ResSEBlock, ConvBlock, GlobalBlock


class Critic(nn.Module):
    def __init__(self, intput_channels, n_blocks, intermediate_channels, config, layer_type = "conv") -> None:
        super(Critic, self).__init__()


        if layer_type == "SE":
            layer = ResSEBlock
        elif layer_type == "conv":
            layer = ConvBlock
        else:
            raise ValueError(f"{layer_type} is not a valid layer type")
        
        blocks = []
        blocks.append(nn.Conv2d(intput_channels, intermediate_channels, kernel_size = 3, padding = 1))
        for _ in range(n_blocks-2):
            blocks.append(layer(intermediate_channels, intermediate_channels))
        blocks.append(layer(intermediate_channels, intermediate_channels))
        blocks.append(nn.Conv2d(intermediate_channels, 5, 1))

        self.global_block = GlobalBlock(config['map_size'], config)

        self.conv = nn.Sequential(*blocks)
        
        self.linear = nn.Sequential(nn.Linear((config['map_size']**2)*5, 1))

    def forward(self, image_features: torch.Tensor, global_features: torch.Tensor):
        image_features = image_features.float()
        global_features = global_features.float()
        global_image_channels = self.global_block(global_features)
        image_features = torch.concatenate((image_features, global_image_channels), dim=1)  # Assumning Batch_Size x Channels x 48 x 48

        image_features = self.conv(image_features)
        image_features = self.linear(image_features.flatten(1))
        return image_features.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)