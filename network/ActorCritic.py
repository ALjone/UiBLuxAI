import torch.nn as nn
from .actor import actor
#from .actor_experimental import actor
from .critic import critic

class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, config):
        super(ActorCritic, self).__init__()
        self.device = config["device"]

        self.channels = 33
        self.actor = actor(self.channels, unit_action_dim, config["actor_n_blocks"], config["actor_intermediate_channels"]).to(self.device)
        # critic
        self.critic = critic(self.channels, config["critic_n_blocks"], config["critic_intermediate_channels"]).to(self.device)


        print("Actor has:", self.actor.count_parameters(), "parameters")
        print("Critic has:", self.critic.count_parameters(), "parameters")
    
    def forward_actor(self, *args):
        return self.actor(*args)

    def forward_critic(self, *args):
        return self.critic(*args)