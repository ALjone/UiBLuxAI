from torch import nn
import torch
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Average feature planes
        y = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5, reduction: int = 4, activation = nn.GELU()):
        """A copy of the conv block from last years winner. Reduction is how many times to reduce the size in the SE"""
        super().__init__()
        assert kernel_size%2 == 1 #Need kernel size to be odd in order to preserve size
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        
        self.SE = SELayer(out_channels, reduction)

        self.activation = activation

    def forward(self, x):
        y = self.activation(self.conv0(x))
        y = self.conv1(y)
        y = self.SE(y)

        x = y + x

        return self.activation(x)
    

class DoubleConv(nn.Module):
    def __init__(self, input_channels, intermediate_channels, n_res_blocks, n_cone_blocks, kernel_size) -> None:
        super().__init__()

        self.activation = nn.GELU()
        padding=(kernel_size-1)//2
        self.conv = nn.Conv2d(input_channels, out_channels=intermediate_channels, kernel_size = kernel_size, padding = padding)

        self.res_part_1 = nn.Sequential(*[ConvBlock(intermediate_channels, intermediate_channels, kernel_size) for _ in range(n_res_blocks)])

        self.double_cone_block = nn.Sequential(*list(
                                                [nn.Conv2d(intermediate_channels, intermediate_channels, 4, stride = 4, padding=padding), self.activation] +
                                                [ConvBlock(intermediate_channels, intermediate_channels, kernel_size) for _ in range(n_cone_blocks)] +
                                                [nn.ConvTranspose2d(intermediate_channels, intermediate_channels, 2, stride = 2), 
                                                 self.activation, 
                                                 nn.ConvTranspose2d(intermediate_channels, intermediate_channels, 2, stride = 2), 
                                                 self.activation]))


        self.res_part_2 = nn.Sequential(*[ConvBlock(intermediate_channels, intermediate_channels, kernel_size) for _ in range(n_res_blocks)])


    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.activation(self.conv(x))

        x = self.res_part_1(x)

        x = self.double_cone_block(x) + x
        return self.res_part_2(x)

class ResNet(nn.Module):
    def __init__(self, kernel_size, input_channels, intermediate_channels, n_blocks) -> None:
        super().__init__()

        layer = ConvBlock

        padding = 1 if kernel_size == 3 else 2

        blocks = []

        #Make shared part
        blocks.append(nn.Conv2d(input_channels, intermediate_channels, kernel_size=kernel_size, padding = padding))
        blocks.append(nn.LeakyReLU())
        for _ in range(n_blocks):
            blocks.append(layer(intermediate_channels, intermediate_channels, kernel_size=kernel_size))

        #Make global features part

        self.conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv(x)