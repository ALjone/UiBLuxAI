from typing import Tuple
import torch
from torch import nn
from .blocks import ResSEBlock, ConvBlock, GlobalBlock


class actor(nn.Module):
    def __init__(self, intput_channels, unit_action_space:int, n_blocks:int, intermediate_channels:int, layer_type = "conv", activation = nn.LeakyReLU()) -> None:
        super(actor, self).__init__()
        
        if layer_type == "SE":
            layer = ResSEBlock
        elif layer_type == "conv":
            layer = ConvBlock
        else:
            raise ValueError(f"{layer_type} is not a valid layer type")
        
        blocks = []

        #Make shared part
        blocks.append(nn.Conv2d(intput_channels, intermediate_channels, kernel_size=5, padding = 2))
        blocks.append(activation)
        for _ in range(n_blocks-2):
            blocks.append(layer(intermediate_channels, intermediate_channels, kernel_size=5))

        blocks.append(nn.Conv2d(intermediate_channels, unit_action_space, 1))

        self.conv = nn.Sequential(*blocks)

    def forward(self, image_features: torch.Tensor, global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(image_features.shape) == 3:
            image_features = image_features.unsqueeze(0)
        if len(global_features.shape) == 1:
            global_features = global_features.unsqueeze(0)

        global_image_channels = global_features.unsqueeze(dim = -1).unsqueeze(dim = -1).repeat(1,1,48,48)
        image_features = torch.concatenate((image_features, global_image_channels), dim=1)  # Assumning Batch_Size x Channels x 48 x 48

        # TODO: 12 new dimensions for image_features input
        x_unit = self.conv(image_features)

        return x_unit.permute(0, 2, 3, 1)
        

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)