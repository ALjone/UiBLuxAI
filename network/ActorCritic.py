import torch.nn as nn
from .actor import Actor
#from .actor_experimental import actor
from .critic import Critic

class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, config):
        super(ActorCritic, self).__init__()
        self.device = config["device"]

        self.channels = 33
        self.actor = Actor(self.channels, unit_action_dim, config["actor_n_blocks"], config["actor_intermediate_channels"], kernel_size = config["kernel_size"]).to(self.device)
        self.critic = Critic(self.channels, intermediate_channels=config["actor_intermediate_channels"], n_blocks = config["critic_n_blocks"], kernel_size = config["kernel_size"]).to(self.device)

    def forward_actor(self, *args):
        return self.actor(*args)
    
    def forward_critic(self, *args):
        return self.critic(*args)
    
    def count_actor_parameters(self):
        return self.actor.count_parameters()
    
    def count_critic_parameters(self):
        return self.critic.count_parameters()