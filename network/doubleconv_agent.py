import torch.nn as nn
from .blocks import DoubleConv
from actions.actions import UNIT_ACTION_IDXS
import torch



class actor(nn.Module):
    def __init__(self, intermediate_channels, kernel_size) -> None:
        super().__init__()

        padding=(kernel_size-1)//2
        self.light_unit_head = nn.Conv2d(intermediate_channels, UNIT_ACTION_IDXS, kernel_size, padding=padding)
        self.heavy_unit_head = nn.Conv2d(intermediate_channels, UNIT_ACTION_IDXS, kernel_size, padding=padding)
        self.factory_head = nn.Conv2d(intermediate_channels, 4, kernel_size, padding=padding)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.factory_head(x).permute(0, 2, 3, 1), self.light_unit_head(x).permute(0, 2, 3, 1), self.heavy_unit_head(x).permute(0, 2, 3, 1)

class critic(nn.Module):
    def __init__(self, kernel_size, intermediate_channels) -> None:
        super().__init__()

        self.activation = nn.GELU()
        self.network = nn.Sequential(nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size, stride=2),
                                     self.activation,
                                     nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size, stride=2),
                                     self.activation,
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(1),
                                     nn.Linear(intermediate_channels, intermediate_channels),
                                     self.activation,
                                     nn.Linear(intermediate_channels, 1))
        
    def forward(self, x):
        return self.network(x).squeeze()


class double_conv_agent(nn.Module):
    def __init__(self, config):
        super(double_conv_agent, self).__init__()
        self.device = config["device"]

        self.channels = 33
        self.backbone = DoubleConv(self.channels, config["actor_intermediate_channels"], config["n_res_blocks"], config["n_cone_blocks"], config["kernel_size"])
        self.actor = actor(config["actor_intermediate_channels"], config["kernel_size"])
        self.critic = critic(config["kernel_size"], config["critic_intermediate_channels"])

        self.backbone.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
       
    def forward(self, x):
        x = self.backbone(x)

        factory_logits, light_unit_logits, heavy_unit_logits = self.actor(x)
        value = self.critic(x)

        return factory_logits, light_unit_logits, heavy_unit_logits, value

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    