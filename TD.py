from functools import partial
import torch
from network.actor import Actor
from actions.old.idx_to_lux_move import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from actions.jux_action_masking import batched_action_mask, unit_action_mask
from jux.torch import to_torch
import time

from torch.nn import MSELoss
class TD:
    def __init__(self, config) -> None:
        self.device = config["device"]
        self.model = Actor(36, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, config["actor_n_blocks"], config["actor_n_blocks_after_split"], config["actor_intermediate_channels"], config).to(self.device)

        self.other_model = Actor(36, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, config["actor_n_blocks"], config["actor_n_blocks_after_split"], config["actor_intermediate_channels"], config).to(self.device)
        self.other_model.load_state_dict(self.model.state_dict())
        
        self.gamma = config["gamma"]
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["regularization"])

        self.action_space = UNIT_ACTION_IDXS

        self.batch_size = config["batch_size"]

        self.loss_func = MSELoss()

        print("Actor has:", self.model.count_parameters(), "parameters")

    def load(self, path):
        pass

    def _unit_mask_actions(self, outputs, state, player_id):
        #1 - player_id to get opponent id
        return outputs
        mask = batched_action_mask(state, player_id)
        outputs[mask] = -torch.inf
        return outputs

    def _factories_mask_actions(self, outputs, state, player_id):
        return outputs
        raise NotImplementedError()
    
    def predict(self, state, image_features, global_features, player_id):
        with torch.no_grad():
            self.model.eval()
            pred_units, pred_factories = self.model(image_features, global_features)
            unit_actions = self._unit_mask_actions(pred_units, state, player_id)
            factories_actions = self._factories_mask_actions(pred_factories, state, player_id)

            return torch.argmax(unit_actions, dim = 1), torch.argmax(factories_actions, dim = 1) #NOTE: Assumes channel first

    def print_size(self, tensor):
         print(self.get_size(tensor))

    def get_size(self, tensor):
        return tensor.nelement()*tensor.element_size()
    
    
    def train2(self, image_states, global_states, unit_actions, factory_actions, rewards, next_image_states, next_global_states, dones, state, player_id):
        #TODO: Mixed precision
        #TODO: Time different parts of training

        self.model.train()

        image_states = to_torch(image_states)
        global_states = to_torch(global_states)
        next_image_states = to_torch(next_image_states)
        next_global_states = to_torch(next_global_states)
        dones = to_torch(dones).to(torch.bool)
        rewards = to_torch(rewards)
        #Player ID, because the unit masks are 0/1
        state_irrelevant_unit_mask = image_states[:, player_id].to(torch.bool)
        state_irrelevant_factory_mask = image_states[:, 2+player_id].to(torch.bool)
        
        next_state_irrelevant_unit_mask = next_image_states[:, player_id].to(torch.bool)
        next_state_irrelevant_factory_mask = next_image_states[:, 2+player_id].to(torch.bool)
        
        num_units = state_irrelevant_unit_mask.sum((1, 2))
        next_num_units = next_state_irrelevant_unit_mask.sum((1, 2))
        num_factories = state_irrelevant_factory_mask.sum((1, 2))

        self.optim.zero_grad()


        Q = self.model(image_states, global_states)
        with torch.no_grad():
            Q_next = self.other_model(next_image_states, next_global_states)

        Q_unit = Q[0]*state_irrelevant_unit_mask.unsqueeze(1)
        Q_factory = Q[1]*state_irrelevant_factory_mask.unsqueeze(1)
        Q_next_unit = self._unit_mask_actions(Q_next[0], state, player_id).max(dim=1).values*next_state_irrelevant_unit_mask
        Q_next_factory = self._factories_mask_actions(Q_next[1], state, player_id).max(dim=1).values*next_state_irrelevant_factory_mask


        """if (next_num_units > 0).any() and (num_units > 0).any():
            loss += torch.square(next_state_irrelevant_unit_mask*Q_next_unit -\
                                    state_irrelevant_unit_mask*Q_unit.gather(dim = 1, index = unit_actions.unsqueeze(1)).squeeze())
        print("\n\n", state_irrelevant_factory_mask*Q_factory.gather(dim = 1, index = factory_actions.unsqueeze(1)).squeeze().shape)
        print((next_state_irrelevant_factory_mask*Q_next_factory).shape)
        loss += torch.square(next_state_irrelevant_factory_mask*Q_next_factory - \
                                state_irrelevant_factory_mask*Q_factory.gather(dim = 1, index = factory_actions.unsqueeze(1)).squeeze())#[torch.arange(factory_actions.shape[0]), factory_actions]) 
        loss = loss.mean()
        #print(loss.item())
        loss.backward()
        self.optim.step()

        return loss.item(), 0, 0, 0, 0, 0, 0, 0"""
        loss_units = 0
        loss_factories = 0
        arange = torch.arange(len(unit_actions[:, 0, 0]))
        #TODO: Add dones
        for i in range(48):
            for j in range(48):
                if (next_num_units > 0).any() and (num_units > 0).any():
                    #print("\n\n")
                    #print(rewards.shape)
                    #print(Q_next_unit[:, i, j].shape)
                    #print(Q_unit[:, :, i, j][torch.arange(len(unit_actions[:, i, j])), unit_actions[:, i, j].flatten()].shape)
                    loss_units += (rewards + Q_next_unit[:, i, j] - Q_unit[:, :, i, j][arange, unit_actions[:, i, j].flatten()]) 
                loss_factories += (rewards + Q_next_factory[:, i, j] - Q_factory[:, :, i, j][arange, factory_actions[:, i, j].flatten()]) 
        
        if (next_num_units > 0).any() and (num_units > 0).any():
            loss_units[num_units > 0] = loss_units[num_units > 0]/num_units[num_units > 0]
            loss_units = torch.abs(loss_units).mean()
            #loss_units.backward()
        loss_factories[num_factories > 0] = loss_factories[num_factories > 0]/num_factories[num_factories > 0]
        loss_factories = torch.abs(loss_factories).mean()

        loss = (loss_units+loss_factories)
        loss.backward()
        #loss = loss.mean()
        #loss.backward()
        self.optim.step()

        return (loss_units+loss_factories).item(), 0, (Q_next_unit + Q_next_factory).mean().item(), rewards.mean().item(), Q_unit.mean().item(), Q_factory.mean().item(), 0, 0


    def train(self, image_states, global_states, unit_actions, factory_actions, rewards, next_image_states, next_global_states, dones, state, player_id):
        
        #TODO: Mixed precision
        #TODO: Time different parts of training


        ###########################################0.1%############################################
        self.model.train()

        image_states = to_torch(image_states)
        global_states = to_torch(global_states)
        next_image_states = to_torch(next_image_states)
        next_global_states = to_torch(next_global_states)
        dones = to_torch(dones).to(torch.bool)
        rewards = to_torch(rewards)
        #Player ID, because the unit masks are 0/1
        state_irrelevant_unit_mask = image_states[:, player_id].to(torch.bool)
        state_irrelevant_factory_mask = image_states[:, 2+player_id].to(torch.bool)
        
        next_state_irrelevant_unit_mask = next_image_states[:, player_id].to(torch.bool)
        next_state_irrelevant_factory_mask = next_image_states[:, 2+player_id].to(torch.bool)
        
        num_units = state_irrelevant_unit_mask.sum((1, 2))
        next_num_units = next_state_irrelevant_unit_mask.sum((1, 2))
        num_factories = state_irrelevant_factory_mask.sum((1, 2))
        next_num_factories = next_state_irrelevant_factory_mask.sum((1, 2))

        self.optim.zero_grad()
        ##############################################################################################


        ###########################################49.51%############################################
        with torch.no_grad():
            Q_next = self.model(next_image_states, next_global_states)
            Q_units_next = torch.zeros(image_states.shape[0], device = self.device)

            #TODO: Should all these values be the sum or the mean? I.e should we actually divide by next_num_units
            if (next_num_units > 0).any():
                Q_units_next[next_num_units > 0] = (self._unit_mask_actions(Q_next[0], state, player_id)[next_num_units > 0].max(dim=1).values*next_state_irrelevant_unit_mask[next_num_units > 0]).sum((1, 2))/next_num_units[next_num_units > 0]

            Q_factories_next = (self._factories_mask_actions(Q_next[1], state, player_id).max(dim=1).values*next_state_irrelevant_factory_mask).sum((1, 2))/next_num_factories
            Q_next = (Q_units_next + Q_factories_next)

        Q_target = rewards + (self.gamma * (~dones)) * Q_next
        ##############################################################################################

        ###########################################35.38%%############################################
        Q = self.model(image_states, global_states) #NOTE: This is 0.05%


        #NOTE: All this below is 35.33%
        gather_time = time.time()
        Q_units = torch.gather(Q[0], dim = 1, index=unit_actions.unsqueeze(1)).squeeze()
        Q_factories = torch.gather(Q[1], dim = 1, index=factory_actions.unsqueeze(1)).squeeze()
        gather_time = time.time()-gather_time

        backprop_time = time.time()
        Q_current_unit = torch.zeros(Q_units.shape[0], device = self.device)
        mask = num_units > 0
        Q_current_unit[mask] = (Q_units[mask]*state_irrelevant_unit_mask[mask]).sum((1, 2))/num_units[mask]
        backprop_time = time.time()-backprop_time

        Q_current_factory = (Q_factories*state_irrelevant_factory_mask).sum((1, 2))/num_factories

        ##############################################################################################

        ############################################15.26%############################################
        #NOTE: Time to compute loss is negligeble
        #print("\n\nTarget:", Q_target[mask])
        #print("\nUnits:", Q_current[mask])
        #print("\nFactories:", (Q_factories*state_irrelevant_factory_mask).sum((1, 2))/num_factories)
        loss = 0
        #if (mask).any():
        #    loss += self.loss_func(Q_current_unit[mask], Q_target[mask])
        loss += self.loss_func(Q_current_factory, Q_target)
        
        loss.backward()
        size = 0
        """        for p1 in self.model.parameters():
            size += self.get_size(p1.grad)
        print("\nSize after forward pass:", size/10**6, "megabyte")"""
        self.optim.step()
        print("\n\n")
        print("Q_target shape:", Q_target.shape)
        print("Q_next shape:", Q_next.shape)
        print("Rewards shape:", rewards.shape)
        print("Q current unit shape:", Q_current_unit.shape)
        print("Q current factory shape_", Q_current_factory.shape)
        print("Loss shape:", loss.shape)
        ##############################################################################################
        return loss.item(), Q_target.mean().item(), Q_next.mean().item(), rewards.mean().item(), Q_current_unit.mean().item(), Q_current_factory.mean().item(), backprop_time, gather_time