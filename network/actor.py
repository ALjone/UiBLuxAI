import torch
from torch import nn
from torchvision.ops import SqueezeExcitation
import numpy as np
from torch.distributions.categorical import Categorical


class actor(nn.Module):
    def __init__(self, intput_channels, output_channels, n_blocks = 10, squeeze_channels = 64) -> None:
        super(actor, self).__init__()

        #TODO: Add split for factory and unit
        
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

    def get_action_and_value(self, x, action = None):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action.shape, probs.log_prob(action).shape, probs.entropy().shape

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)