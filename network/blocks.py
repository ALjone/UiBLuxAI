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

class GlobalBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 12)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # Batch_size x 12 --> Batch_size x 12 x 1 x 1 --> Batch_size x 12 x 48 x 48

        x = x.unsqueeze(dim = -1).unsqueeze(dim = -1)
        x = torch.repeat(1,1,48,48)
        return x


class RotationInvariantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(RotationInvariantConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.bias = nn.Parameter(torch.rand(out_channels), requires_grad=True).to(devise)
        # create a parameter for each symmetric group
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(1, 1, 1, 1)).to(devise) for i in range(6*out_channels)]).to(device=devise)
        self.reset_parameters()

        self.weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, device=devise)
        for i in range(self.out_channels):
            self.weight[i,0,0,0] = self.weights[0 + 6*i]
            self.weight[i,0,-1,0] = self.weights[0+ 6*i]
            self.weight[i,0,0,-1] = self.weights[0+ 6*i]
            self.weight[i,0,-1,-1] = self.weights[0+ 6*i]

            self.weight[i,0,0,2] = self.weights[1+ 6*i]
            self.weight[i,0,2,0] = self.weights[1+ 6*i]
            self.weight[i,0,2,-1] = self.weights[1+ 6*i]
            self.weight[i,0,-1,2] = self.weights[1+ 6*i]

            self.weight[i,0,0,1] = self.weights[2+ 6*i]
            self.weight[i,0,0,3] = self.weights[2+ 6*i]
            self.weight[i,0,1,0] = self.weights[2+ 6*i]
            self.weight[i,0,3,0] = self.weights[2+ 6*i]
            self.weight[i,0,1,-1] = self.weights[2+ 6*i]
            self.weight[i,0,3,-1] = self.weights[2+ 6*i]
            self.weight[i,0,-1,1] = self.weights[2+ 6*i]
            self.weight[i,0,-1,3] = self.weights[2+ 6*i]

            self.weight[i,0,1,1] = self.weights[3+ 6*i]
            self.weight[i,0,1,3] = self.weights[3+ 6*i]
            self.weight[i,0,3,1] = self.weights[3+ 6*i]
            self.weight[i,0,3,3] = self.weights[3+ 6*i]

            self.weight[i,0,1,2] = self.weights[4+ 6*i]
            self.weight[i,0,3,2] = self.weights[4+ 6*i]
            self.weight[i,0,2,1] = self.weights[4+ 6*i]
            self.weight[i,0,2,3] = self.weights[4+ 6*i]

            self.weight[i,0, 2, 2] = self.weights[5 + 6*i]

    def reset_parameters(self):
        n = self.in_channels * 6
        stdv = 1#1. / math.sqrt(n)
        for weight in self.weights:
            weight.data.normal_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, stride=self.stride, padding=self.padding, bias = self.bias)