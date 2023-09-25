import torch.nn as nn
from .actor import actor
#from .actor_experimental import actor
from .critic import critic

class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, config):
        super(ActorCritic, self).__init__()
        self.device = config["device"]

        self.channels = 33
        self.actor = actor(self.channels, unit_action_dim, config["actor_n_blocks"], config["actor_intermediate_channels"], use_batch_norm=config["actor_use_batch_norm"]).to(self.device)
        self.critic = critic(self.channels, intermediate_channels=config["actor_intermediate_channels"]).to(self.device)

    def forward_actor(self, *args):
        return self.actor(*args)
    
    def forward_critic(self, *args):
        return self.critic(*args)
    
    def count_actor_parameters(self):
        return self.actor.count_parameters()
    
    def count_critic_parameters(self):
        return self.critic.count_parameters()