import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical


class critic(nn.Module):
    def __init__(self, intput_channels, output_channels, n_blocks = 10, squeeze_channels = 64) -> None:
        super(critic, self).__init__()

        #TODO: Add split for factory and unit
        
        self.blocks = torch.nn.ParameterList()
        self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        for _ in range(n_blocks-2):
            self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        self.blocks.append(nn.Conv2d(intput_channels, 5, 1))
        
        self.linear = nn.Linear(11520, 1)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()


        for layer in self.blocks:
            x = layer(x) 
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