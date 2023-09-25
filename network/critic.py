import torch
from torch import nn


class critic(nn.Module):
    def __init__(self, intput_channels, intermediate_channels, activation = nn.LeakyReLU()) -> None:
        super(critic, self).__init__()
        
        blocks = []
        blocks.append(nn.Conv2d(intput_channels, intermediate_channels, kernel_size = 5, padding = 1))
        blocks.append(activation)
        for _ in range(6):
            blocks.append(nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=5, padding = 1))
            blocks.append(activation)
        blocks.append(nn.Conv2d(intermediate_channels, 64, kernel_size=1))
        blocks.append(activation)

        self.conv = nn.Sequential(*blocks)

        self.linear = nn.Sequential(nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 1))
        

    def forward(self, image_features: torch.Tensor):
        if len(image_features.shape) == 3:
            image_features = image_features.unsqueeze(0)

        x = self.conv(image_features)
        x = nn.AvgPool2d(x.shape[-2])(x).squeeze()
        x = self.linear(x)
        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)