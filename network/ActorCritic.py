import torch.nn as nn
from .actor import actor
#from .actor_experimental import actor
from .critic import critic

class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, config):
        super(ActorCritic, self).__init__()
        self.device = config["device"]

        self.channels = 29
        self.actor_critic = actor(self.channels, unit_action_dim, config["actor_n_blocks"], config["actor_intermediate_channels"], use_batch_norm=config["actor_use_batch_norm"]).to(self.device)

    def forward_actor(self, *args):
        return self.actor_critic(*args)
    
    def count_parameters(self):
        return self.actor_critic.count_parameters()