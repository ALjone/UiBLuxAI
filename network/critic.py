import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical
from .blocks import ResSEBlock, ConvBlock


class critic(nn.Module):
    def __init__(self, intput_channels, n_blocks = 4, intermediate_channels = 32, layer_type = "conv") -> None:
        super(critic, self).__init__()


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

        self.conv = nn.Sequential(*blocks)
        
        self.linear = nn.Linear((48**2)*5, 1)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()


        x = self.conv(x)
        x = self.linear(x.flatten(1))
        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = critic(23, 10)
    print("Number of parameters in network", net.count_parameters())
    tens = torch.ones((5, 23, 48, 48))

    print(torch.sum(tens))

    print(torch.sum(net(tens)))

    print(net(tens).shape)