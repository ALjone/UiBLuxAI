from audioop import reverse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from network.ActorCritic import ActorCritic

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.unit_actions = []
        self.factory_actions = []
        self.image_features = []
        self.global_features = []
        self.unit_logprobs = []
        self.factory_logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.unit_actions[:]
        del self.factory_actions[:]
        del self.image_features[:]
        del self.global_features[:]
        del self.unit_logprobs[:]
        del self.factory_logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, unit_action_dim, factory_action_dim, config):
        self.device = config["device"]
        self.gamma = config["gamma"]
        self.eps_clip = config["eps_clip"]
        self.K_epochs = config["epochs_per_batch"]
        self.mini_batch_size = config["mini_batch_size"]
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(unit_action_dim, factory_action_dim, config).to(config["device"])
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': config["actor_lr"]},
                        {'params': self.policy.critic.parameters(), 'lr': config["critic_lr"]}
                    ])

        #self.policy_old = ActorCritic(unit_action_dim, factory_action_dim, config).to(config["device"])
        #self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, image_features, global_features, obs):

        with torch.no_grad():
            image_features = image_features.float().to(self.device)
            action_unit, action_factory, action_logprobs_unit, action_logprobs_factories, state_values = self.policy.act(image_features, global_features, obs)
        
        self.buffer.image_features.append(image_features)
        self.buffer.global_features.append(global_features)
        self.buffer.unit_actions.append(action_unit)
        self.buffer.factory_actions.append(action_factory)
        self.buffer.unit_logprobs.append(action_logprobs_unit)
        self.buffer.factory_logprobs.append(action_logprobs_factories)
        self.buffer.state_values.append(state_values)

        return action_unit.squeeze(), action_factory.squeeze()

    def update(self):
        # Monte Carlo estimate of returns
        
        rewards = torch.zeros(len(self.buffer.rewards), dtype = torch.float32)
        discounted_reward = 0
        for i, (reward, is_terminal) in enumerate(zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals))):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards[len(self.buffer.rewards)-i-1] = discounted_reward
            
        # Normalizing the rewards
        rewards = rewards.to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_image_features = torch.squeeze(torch.stack(self.buffer.image_features, dim=0)).detach().to(self.device)
        old_global_features = torch.squeeze(torch.stack(self.buffer.global_features, dim=0)).detach().to(self.device)
        old_actions_unit = torch.squeeze(torch.stack(self.buffer.unit_actions, dim=0)).detach().to(self.device)
        old_actions_factory = torch.squeeze(torch.stack(self.buffer.factory_actions, dim=0)).detach().to(self.device)
        old_logprobs_unit = torch.squeeze(torch.stack(self.buffer.unit_logprobs, dim=0)).detach().to(self.device)
        old_logprobs_factory = torch.squeeze(torch.stack(self.buffer.factory_logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = (rewards.detach() - old_state_values.detach())

        cum_loss = 0
        # Optimize policy for K epochs
        for _ in tqdm(range(self.K_epochs), leave = False, desc = "Training"):

            # Evaluating old actions and values
            mini_batch_indices = np.random.choice(list(range(len(old_image_features))), self.mini_batch_size)
            action_logprobs_unit, action_probs_factories, state_values, unit_dist_entropy, factory_dist_entropy = self.policy.evaluate(old_image_features[mini_batch_indices], 
                                    old_global_features[mini_batch_indices], old_actions_unit[mini_batch_indices], old_actions_factory[mini_batch_indices])
        
            loss = 0
            for logprobs, dist_entropy, old_logprobs in [(action_logprobs_unit, unit_dist_entropy, old_logprobs_unit),
                                                          (action_probs_factories, factory_dist_entropy, old_logprobs_factory)]:
                
                # Finding the ratio (pi_theta / pi_theta__old)

                ratios = torch.exp(logprobs - old_logprobs.detach()[mini_batch_indices])

                # Finding Surrogate Loss  
                surr1 = advantages[mini_batch_indices]*ratios#ratios @ advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[mini_batch_indices]

                # final loss of clipped objective PPO
                loss += -torch.min(surr1, surr2) - 0.01 * dist_entropy
                
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            loss += 0.5 * self.MseLoss(state_values, rewards[mini_batch_indices])

            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            cum_loss += loss.item()
            
        # Copy new weights into old policy
        #self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return cum_loss/self.K_epochs
    
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        #self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

