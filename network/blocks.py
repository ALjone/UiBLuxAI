from torch import nn
from torchvision.ops import SqueezeExcitation
import torch
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, n_channels: int, rescale_input: bool, reduction: int = 16):
        super(SELayer, self).__init__()
        self.rescale_input = rescale_input
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Average feature planes
        if self.rescale_input:
            y = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1)
        else:
            y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5, reduction: int = 16, activation = nn.LeakyReLU()):
        """A copy of the conv block from last years winner. Reduction is how many times to reduce the size in the SE"""
        super().__init__()
        assert kernel_size%2 == 1 #Need kernel size to be odd in order to preserve size
        self.conv0 = RotationInvariantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)#nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv1 = RotationInvariantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)#nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        
        self.SE = SELayer(out_channels, True, reduction)

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
        #self.fc1 = nn.Linear(13, 64)
        #self.fc2 = nn.Linear(64, 13)

        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 12)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # Batch_size x 12 --> Batch_size x 12 x 1 x 1 --> Batch_size x 12 x 48 x 48

        x = x.unsqueeze(dim = -1).unsqueeze(dim = -1)
        x = x.repeat(1,1,48,48)
        return x

class RotationInvariantConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(RotationInvariantConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        assert kernel_size == 5, "Found kernel size " + str(kernel_size)

        self.bias = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
        # create a parameter for each symmetric group
        self.weight = nn.ParameterList([torch.nn.Parameter(torch.Tensor(1, 1, 1, 1), requires_grad=True) for i in range(6*out_channels)])
        #self.reset_parameters()

        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)

        #
        with torch.no_grad():
        # make a copy of self.weights tensor
            new_weights = self.weights.clone()
            for i in range(self.out_channels):
                new_weights[i,0,0,0] = self.weight[0 + 6*i]
                new_weights[i,0,-1,0] = self.weight[0+ 6*i]
                new_weights[i,0,0,-1] = self.weight[0+ 6*i]
                new_weights[i,0,-1,-1] = self.weight[0+ 6*i]

                new_weights[i,0,0,2] = self.weight[1+ 6*i]
                new_weights[i,0,2,0] = self.weight[1+ 6*i]
                new_weights[i,0,2,-1] = self.weight[1+ 6*i]
                new_weights[i,0,-1,2] = self.weight[1+ 6*i]

                new_weights[i,0,0,1] = self.weight[2+ 6*i]
                new_weights[i,0,0,3] = self.weight[2+ 6*i]
                new_weights[i,0,1,0] = self.weight[2+ 6*i]
                new_weights[i,0,3,0] = self.weight[2+ 6*i]
                new_weights[i,0,1,-1] = self.weight[2+ 6*i]
                new_weights[i,0,3,-1] = self.weight[2+ 6*i]
                new_weights[i,0,-1,1] = self.weight[2+ 6*i]
                new_weights[i,0,-1,3] = self.weight[2+ 6*i]

                new_weights[i,0,1,1] = self.weight[3+ 6*i]
                new_weights[i,0,1,3] = self.weight[3+ 6*i]
                new_weights[i,0,3,1] = self.weight[3+ 6*i]
                new_weights[i,0,3,3] = self.weight[3+ 6*i]

                new_weights[i,0,1,2] = self.weight[4+ 6*i]
                new_weights[i,0,3,2] = self.weight[4+ 6*i]
                new_weights[i,0,2,1] = self.weight[4+ 6*i]
                new_weights[i,0,2,3] = self.weight[4+ 6*i]

                new_weights[i,0, 2, 2] = self.weight[5 + 6*i]
    
            # Assign new_weights_list to self.weights
            self.weights = nn.Parameter(new_weights)
        
            self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                param.data.normal_(0, 0.01)

    def forward(self, input):
        #print(np.sum(self.weights.data.detach().cpu()))
        return F.conv2d(input, self.weights, stride=self.stride, padding=self.padding, bias = self.bias)

