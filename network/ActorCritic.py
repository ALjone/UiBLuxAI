import torch
import torch.nn as nn
from torch.distributions import Categorical
from .actor import actor
from .critic import critic


class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, factory_action_dim, device = torch.device("cuda")):
        super(ActorCritic, self).__init__()

        self.actor = actor(23, unit_action_dim, factory_action_dim).to(device)
        # critic
        self.critic = critic(23).to(device)

        print("Actor has:", self.actor.count_parameters(), "parameters")
        print("Critic has:", self.critic.count_parameters(), "parameters")
        
    def forward(self, state):
        return self.act(state)
    
    def act(self, state):
        #TODO: Should these be mean?
        action_probs_unit, action_probs_factories = self.actor(state)
        unit_dist, factory_dist = Categorical(action_probs_unit), Categorical(action_probs_factories)

        action_unit = unit_dist.sample()
        action_logprob_unit = unit_dist.log_prob(action_unit)

        action_factory = factory_dist.sample()
        action_logprob_factory = factory_dist.log_prob(action_factory)

        state_val = self.critic(state)

        return action_unit.detach(), action_factory.detach(), torch.mean(action_logprob_unit.detach()), torch.mean(action_logprob_factory.detach()), state_val.detach()
    
    def evaluate(self, state, unit_action, factory_action):
        #TODO: Should these be mean?
        action_probs_unit, action_probs_factories = self.actor(state)

        unit_dist = Categorical(action_probs_unit)
        action_logprobs_unit = unit_dist.log_prob(unit_action)
        unit_dist_entropy = unit_dist.entropy().mean((1, 2))

        factory_dist = Categorical(action_probs_factories)
        action_logprobs_factories = factory_dist.log_prob(factory_action)
        factory_dist_entropy = factory_dist.entropy().mean((1, 2))

        state_values = self.critic(state)
        
        return torch.mean(action_logprobs_unit, dim=(1, 2)), torch.mean(action_logprobs_factories, dim = (1, 2)), state_values, unit_dist_entropy, factory_dist_entropy