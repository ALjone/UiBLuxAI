import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from actor import actor
from critic import critic


class ActorCritic(nn.Module):
    def __init__(self, state_dim, device = torch.device("cuda")):
        super(ActorCritic, self).__init__()

        self.actor = actor(state_dim).to(device)
        # critic
        self.critic = critic(state_dim).to(device)
        
    def forward(self, state):
        self.act(state)
    
    def act(self, state):


        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):


        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy