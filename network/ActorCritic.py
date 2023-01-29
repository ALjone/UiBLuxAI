import torch
import torch.nn as nn
from torch.distributions import Categorical
from .actor import actor
from .critic import critic


class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, device = torch.device("cuda")):
        super(ActorCritic, self).__init__()

        self.actor = actor(4, unit_action_dim).to(device)
        # critic
        self.critic = critic(4).to(device)
        
    def forward(self, state):
        return self.act(state)
    
    def act(self, state):
        self.actor.eval()
        self.critic.eval()
        action_probs_unit = self.actor(state)
        unit_dist = Categorical(action_probs_unit)

        action_unit = unit_dist.sample()
        action_logprob_unit = unit_dist.log_prob(action_unit)


        state_val = self.critic(state)

        return action_unit.detach(), action_logprob_unit.detach(), state_val.detach()
    
    def evaluate(self, state, unit_action):
        self.actor.train()
        self.critic.train()
        action_probs_unit = self.actor(state)

        unit_dist = Categorical(action_probs_unit)
        action_logprobs_unit = unit_dist.log_prob(unit_action)
        unit_dist_entropy = unit_dist.entropy()

        state_values = self.critic(state)
        return torch.mean(action_logprobs_unit, dim=(1, 2)), torch.mean(action_logprobs_factories, dim = (1, 2)), state_values, unit_dist_entropy, factory_dist_entropy
