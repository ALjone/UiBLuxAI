import torch
import torch.nn as nn
from torch.distributions import Categorical
from .actor import actor
from .critic import critic
from actions.action_masking import unit_action_mask, factory_action_mask

class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, factory_action_dim, config):
        super(ActorCritic, self).__init__()
        self.device = config["device"]

        channels = 31

        self.actor = actor(channels, unit_action_dim, factory_action_dim, config["actor_n_blocks"], config["actor_n_blocks_after_split"], config["actor_intermediate_channels"]).to(self.device)
        # critic
        self.critic = critic(channels, config["critic_n_blocks"], config["critic_intermediate_channels"]).to(self.device)


        print("Actor has:", self.actor.count_parameters(), "parameters")
        print("Critic has:", self.critic.count_parameters(), "parameters")
        
    def forward(self, state):
        return self.act(state)
    
    def act(self, image_features, global_features, obs):
        #NOTE: Assumes first channel is unit mask for our agent
        #NOTE: Assumes second channel is factory mask for our agent
        
        #TODO: Add
        #unit_mask = unit_action_mask(obs, self.device)
        #factory_mask = factory_action_mask(obs, self.device)
        #assert action_probs_unit.shape == unit_mask.shape
        #assert action_probs_factories.shape == factory_mask.shape
        #action_probs_unit *= unit_mask
        #action_probs_factories *= factory_mask
        #action_probs_unit *= unit_mask


        action_probs_unit, action_probs_factories = self.actor(image_features, global_features)
        unit_dist, factory_dist = Categorical(action_probs_unit), Categorical(action_probs_factories)

        action_unit = unit_dist.sample()
        action_factory = factory_dist.sample()

        action_logprob_unit = unit_dist.log_prob(action_unit)*(image_features[0] == 1)
        action_logprob_factory = factory_dist.log_prob(action_factory)*(image_features[1] == 1)

        state_val = self.critic(image_features, global_features)

        return action_unit, action_factory, torch.sum(action_logprob_unit), torch.sum(action_logprob_factory), state_val
    
    def evaluate(self, image_features, global_features, unit_action, factory_action):
        #TODO: Does this also need the same type of action masking? Yes, according to gridnet
        #https://github.com/vwxyzjn/gym-microrts-paper/blob/master/ppo_gridnet_diverse_impala.py Line number 342

        #NOTE: Assumes first channel is unit mask for our agent
        #NOTE: Assumes second channel is factory mask for our agent
        action_unit_probs, action_probs_factories = self.actor(image_features, global_features)

        #TODO: Fix. This needs obs, or just straight up the masks 
        """unit_mask = unit_action_mask(obs, self.device)
        factory_mask = factory_action_mask(obs, self.device)
        action_probs_factories *= factory_mask
        action_probs_unit *= unit_mask"""


        unit_dist = Categorical(action_unit_probs)
        factory_dist = Categorical(action_probs_factories)


        action_logprobs_unit = unit_dist.log_prob(unit_action)*(image_features[:, 0] == 1)
        action_logprobs_factories = factory_dist.log_prob(factory_action)*(image_features[:, 1] == 1)


        unit_dist_entropy = (unit_dist.entropy()*image_features[:, 0]).sum((1, 2))
        factory_dist_entropy = (factory_dist.entropy()*image_features[:, 1]).sum((1, 2))

        state_values = self.critic(image_features, global_features)
        
        return action_logprobs_unit.sum((1, 2)), action_logprobs_factories.sum((1, 2)), state_values, unit_dist_entropy, factory_dist_entropy