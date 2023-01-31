from audioop import reverse
import torch
import torch.nn as nn
from tqdm import tqdm
from network.ActorCritic import ActorCritic

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.unit_actions = []
        self.factory_actions = []
        self.states = []
        self.unit_logprobs = []
        self.factory_logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.unit_actions[:]
        del self.factory_actions[:]
        del self.states[:]
        del self.unit_logprobs[:]
        del self.factory_logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, unit_action_dim, factory_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device = torch.device("cuda")):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(unit_action_dim, factory_action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(unit_action_dim, factory_action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        with torch.no_grad():
            state = state.float().to(self.device)
            action_unit, action_factory, action_logprobs_unit, action_logprobs_factories, state_values = self.policy_old.act(state)
        
        self.buffer.states.append(state)
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
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions_unit = torch.squeeze(torch.stack(self.buffer.unit_actions, dim=0)).detach().to(self.device)
        old_actions_factory = torch.squeeze(torch.stack(self.buffer.factory_actions, dim=0)).detach().to(self.device)
        old_logprobs_unit = torch.squeeze(torch.stack(self.buffer.unit_logprobs, dim=0)).detach().to(self.device)
        old_logprobs_factory = torch.squeeze(torch.stack(self.buffer.factory_logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = (rewards.detach() - old_state_values.detach())

        #TODO: Check if it makes sense to loop over probs and entropy when training
        cum_loss = 0
        # Optimize policy for K epochs
        for _ in tqdm(range(self.K_epochs), leave = False, desc = "Training"):

            # Evaluating old actions and values
            action_logprobs_unit, action_probs_factories, state_values, unit_dist_entropy, factory_dist_entropy = self.policy.evaluate(old_states, old_actions_unit, old_actions_factory)

            loss = 0

            for logprobs, dist_entropy, old_logprobs in [(action_logprobs_unit, unit_dist_entropy, old_logprobs_unit),
                                                          (action_probs_factories, factory_dist_entropy, old_logprobs_factory)]:
                
                # Finding the ratio (pi_theta / pi_theta__old)

                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss  
                surr1 = advantages*ratios#ratios @ advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss += -torch.min(surr1, surr2) - 0.01 * dist_entropy
                
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            loss += 0.5 * self.MseLoss(state_values, rewards)

            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            cum_loss += loss.item()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return cum_loss/self.K_epochs
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

