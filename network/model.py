import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np

class actor(nn.Module):
    def __init__(self, intput_channels, output_channels, n_blocks = 10, squeeze_channels = 64) -> None:
        super(actor, self).__init__()
        
        self.blocks = torch.nn.ParameterList()
        self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        for _ in range(n_blocks-2):
            self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        self.blocks.append(SqueezeExcitation(intput_channels, squeeze_channels))
        self.blocks.append(nn.Conv2d(intput_channels, output_channels, 1))

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()


        for layer in self.blocks:
            x = layer(x) 
        return x.squeeze().permute(1, 2, 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = actor(23, 10)
    print("Number of parameters in network", net.count_parameters())
    tens = torch.ones((1, 23, 48, 48))

    print(torch.sum(tens))

    print(torch.sum(net(tens)))