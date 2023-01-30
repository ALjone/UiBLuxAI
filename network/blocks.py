import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation


class ResConvBlock(nn.Module):
    def __init__(self, channels, intermediate_channels ,kernel_size = 3, activation_function = nn.ReLU()):
        super().__init__()
        assert channels == intermediate_channels
        assert kernel_size%2 == 1 #Need kernel size to be odd in order to preserve size
        self.activation_function = activation_function
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        inputs = x
        x = self.activation_function(x)
        x = self.conv0(x)
        x = self.activation_function(x)
        x = self.conv1(x)
        return x + inputs
    


class ResSEBlock(nn.Module):
    def __init__(self, input_channels, squeeze_channels, kernel_size = 3, activation_function = nn.ReLU()):
        super().__init__()
        assert kernel_size%2 == 1 #Need kernel size to be odd in order to preserve size
        self.activation_function = activation_function
        self.SE0 = SqueezeExcitation(input_channels, squeeze_channels)
        self.SE1 = SqueezeExcitation(input_channels, squeeze_channels)

    def forward(self, x):
        inputs = x
        x = self.activation_function(x)
        x = self.SE0(x)
        x = self.activation_function(x)
        x = self.SE1(x)
        return x + inputs