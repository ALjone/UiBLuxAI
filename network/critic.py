import torch
from torch import nn
from .blocks import ResSEBlock, ConvBlock, GlobalBlock, RotationInvariantConv2d


class critic(nn.Module):
    def __init__(self, intput_channels, n_blocks, intermediate_channels, activation = nn.LeakyReLU()) -> None:
        super(critic, self).__init__()
        
        blocks = []
        blocks.append(RotationInvariantConv2d(intput_channels, intermediate_channels, kernel_size = 5, padding = 1))
        blocks.append(activation)
        blocks.append(nn.BatchNorm2d(intermediate_channels))
        for _ in range(10):
            blocks.append(RotationInvariantConv2d(intermediate_channels, intermediate_channels, kernel_size=5))
            blocks.append(activation)
            blocks.append(nn.BatchNorm2d(intermediate_channels))
        blocks.append(RotationInvariantConv2d(intermediate_channels, 128, kernel_size=5, stride = 2))
        blocks.append(activation)
        blocks.append(nn.BatchNorm2d(128))

        self.conv = nn.Sequential(*blocks)

        self.global_block =  GlobalBlock()
        
        self.linear = nn.Sequential(nn.Linear(128, 128),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(64),
                                    nn.Linear(64, 1))

    def forward(self, image_features: torch.Tensor):
        raise NotImplementedError("Class not supported")
        if len(image_features.shape) == 3:
            image_features = image_features.unsqueeze(0)

        image_features = self.conv(image_features)
        image_features = self.linear(image_features.flatten(1))
        return image_features.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
