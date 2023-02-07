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

        channels = 32

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
        
        #unit_mask = unit_action_mask(obs, self.device)
        factory_mask = factory_action_mask(obs, self.device)


        action_type_probs, action_direction_probs, action_values_probs, action_probs_factories = self.actor(image_features, global_features)
        print(action_type_probs.shape)
        print(action_direction_probs.shape)
        print(action_values_probs.shape)
        #assert action_probs_unit.shape == unit_mask.shape
        assert action_probs_factories.shape == factory_mask.shape

        #action_probs_unit *= unit_mask
        action_probs_factories *= factory_mask
        type_dist, direction_dist, value_dist, factory_dist = Categorical(action_type_probs), Categorical(action_direction_probs), Categorical(action_values_probs), Categorical(action_probs_factories)

        action_type = type_dist.sample()
        action_direction = direction_dist.sample()
        action_value = value_dist.sample()

        unit_mask = (image_features[0] == 1)
        action_type_logprob = type_dist.log_prob(action_type)*unit_mask
        action_direction_logprob = direction_dist.log_prob(action_direction)*unit_mask
        action_value_logprob = value_dist.log_prob(action_value)*unit_mask

        # Current problem: How do we handle logprobs when we are sampling 3 values from 3 different (not independent) distributions
        action_logprob_unit = torch.sum(action_type_logprob.detach()) * torch.sum(action_direction_logprob.detach()) * torch.sum(action_value_logprob.detach())

        action_factory = factory_dist.sample()
        action_logprob_factory = factory_dist.log_prob(action_factory)*(image_features[1] == 1)

        state_val = self.critic(image_features, global_features)

        return action_type.detach(), action_direction.detach(), action_value.detach(), \
            action_factory.detach(), torch.sum(action_logprob_unit.detach()), torch.sum(action_logprob_factory.detach()), state_val.detach()
    
    def evaluate(self, image_features, global_features, unit_action, factory_action):
        #TODO: Does this also need the same type of action masking? Yes, according to gridnet
        #https://github.com/vwxyzjn/gym-microrts-paper/blob/master/ppo_gridnet_diverse_impala.py Line number 342

        #NOTE: Assumes first channel is unit mask for our agent
        #NOTE: Assumes second channel is factory mask for our agent
        action_probs_unit, action_probs_factories = self.actor(image_features, global_features)

        unit_dist = Categorical(action_probs_unit)
        action_logprobs_unit = unit_dist.log_prob(unit_action)*(image_features[:, 0] == 1)
        unit_dist_entropy = (unit_dist.entropy()*image_features[:, 0]).sum((1, 2))

        factory_dist = Categorical(action_probs_factories)
        action_logprobs_factories = factory_dist.log_prob(factory_action)*(image_features[:, 1] == 1)
        factory_dist_entropy = (factory_dist.entropy()*image_features[:, 0]).sum((1, 2))

        state_values = self.critic(image_features, global_features)
        
        return torch.sum(action_logprobs_unit, dim=(1, 2)), torch.sum(action_logprobs_factories, dim = (1, 2)), state_values, unit_dist_entropy, factory_dist_entropy