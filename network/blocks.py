import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5, reduction: int = 16, activation = nn.LeakyReLU()):
        """A copy of the conv block from last years winner. Reduction is how many times to reduce the size in the SE"""
        super().__init__()
        assert kernel_size%2 == 1 #Need kernel size to be odd in order to preserve size
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        
        self.SE = SqueezeExcitation(out_channels, out_channels//reduction)

        self.activation = activation
        
        if in_channels != out_channels:
            self.change_channels = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.change_channels = lambda x: x

    def forward(self, x):
        pre = x
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.SE(x))

        x = x + self.change_channels(pre)

        return self.activation(x)

class ResConvBlock(nn.Module):
    def __init__(self, channels, kernel_size = 3, activation_function = nn.ReLU()):
        super().__init__()
        assert kernel_size%2 == 1 #Need kernel size to be odd in order to preserve size
        self.activation_function = activation_function
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        inputs = x
        x = self.conv0(x)
        x = self.activation_function(x)
        x = self.conv1(x)
        x = self.activation_function(x)
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
        x = self.SE0(x)
        x = self.activation_function(x)
        x = self.SE1(x)
        x = self.activation_function(x)
        return x + inputs