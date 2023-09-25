import torch
import torch.nn as nn
from torch.distributions import Categorical
from .actor import Actor
from .critic import Critic
from actions.lux_action_masking import unit_action_mask, factory_action_mask

class ActorCritic(nn.Module):
    def __init__(self, unit_action_dim, factory_action_dim, config):
        super(ActorCritic, self).__init__()
        self.device = config["device"]

        channels = 32

        self.actor = Actor(channels, unit_action_dim, factory_action_dim, config["actor_n_blocks"], config["actor_n_blocks_after_split"], config["actor_intermediate_channels"], config).to(self.device)
        # critic
        self.critic = Critic(channels, config["critic_n_blocks"], config["critic_intermediate_channels"], config).to(self.device)


        print("Actor has:", self.actor.count_parameters(), "parameters")
        print("Critic has:", self.critic.count_parameters(), "parameters")
        
    def forward(self, state):
        return self.act(state)
    
    def act(self, image_features, global_features, obs, player_idx):
        #NOTE: Assumes first channel is unit mask for our agent
        #NOTE: Assumes second channel is factory mask for our agent
        
        #unit_mask = unit_action_mask(obs, self.device)
        unit_mask = unit_action_mask(obs, self.device, player_idx)
        factory_mask = factory_action_mask(obs, self.device, player_idx)

        if len(image_features.shape) == 3:
            image_features = image_features.unsqueeze(0)
        action_unit_probs, action_probs_factories = self.actor(image_features, global_features)
        #assert action_probs_unit.shape == unit_mask.shape

        assert action_probs_factories.shape == factory_mask.shape

        #action_probs_unit *= unit_mask
        action_probs_factories[factory_mask] = 0

        action_unit_probs[unit_mask] = 0
        unit_dist, factory_dist = Categorical(logits = action_unit_probs.permute(0, 2, 3, 1)), Categorical(logits = action_probs_factories.permute(0, 2, 3, 1))
        action_unit = unit_dist.sample()
        irrelevant_unit_mask = (image_features[:, 0] == 1)
        irrelevant_factory_mask = (image_features[:, 2] == 1)
        action_unit_logprob = unit_dist.log_prob(action_unit)*irrelevant_unit_mask
        
        action_factory = factory_dist.sample()
        action_logprob_factory = factory_dist.log_prob(action_factory)*irrelevant_factory_mask

        state_val = self.critic(image_features, global_features)

        return action_unit.detach(), action_factory.detach(), torch.sum(action_unit_logprob.detach(), dim = (1,2)), torch.sum(action_logprob_factory.detach(), dim = (1,2)), state_val.detach()
    
    def evaluate(self, image_features, global_features, unit_action, factory_action):
        #TODO: Does this also need the same type of action masking? Yes, according to gridnet
        #https://github.com/vwxyzjn/gym-microrts-paper/blob/master/ppo_gridnet_diverse_impala.py Line number 342

        #NOTE: Assumes first channel is unit mask for our agent
        #NOTE: Assumes third channel is factory mask for our agent
        action_unit_probs, action_probs_factories = self.actor(image_features, global_features)

        #unit_dist = Categorical(action_probs_unit)
        unit_dist = Categorical(logits = action_unit_probs.permute(0, 2, 3, 1))

        irrelevant_unit_mask = (image_features[:,0] == 1)
        irrelevant_factory_mask = (image_features[:,2] == 1)
        action_unit_logprob = unit_dist.log_prob(unit_action)*irrelevant_unit_mask

        # Current problem: How do we handle logprobs when we are sampling 3 values from 3 different (not independent) distributions
        # TODO: not sure about how to add these entropies
        unit_dist_entropy = (unit_dist.entropy()*irrelevant_unit_mask).sum((1, 2))

        factory_dist = Categorical(logits = action_probs_factories.permute(0, 2, 3, 1))
        action_logprobs_factories = factory_dist.log_prob(factory_action)*irrelevant_factory_mask
        factory_dist_entropy = (factory_dist.entropy()*irrelevant_factory_mask).sum((1, 2))

        state_values = self.critic(image_features, global_features)
        
        return torch.sum(action_unit_logprob, dim=(1, 2)), torch.sum(action_logprobs_factories, dim = (1, 2)), state_values, unit_dist_entropy, factory_dist_entropy