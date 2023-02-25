import torch
from network.actor import Actor
from actions.idx_to_lux_move import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from actions.action_masking import batched_action_mask, unit_action_mask
from jux.torch import to_torch
class TD:
    def __init__(self, config) -> None:
        self.device = config["device"]
        self.model  = Actor(36, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, config["actor_n_blocks"], config["actor_n_blocks_after_split"], config["actor_intermediate_channels"], config).to(self.device)
        
        self.gamma = config["gamma"]
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["regularization"])

        self.action_space = UNIT_ACTION_IDXS

        self.batch_size = config["batch_size"]
        print("Actor has:", self.model.count_parameters(), "parameters")

    def _unit_mask_actions(self, outputs, state, player_id):
        #1 - player_id to get opponent id
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

    def get_size(self, tensor):
         print(tensor.nelement()*tensor.element_size())

    def train(self, image_states, global_states, unit_actions, factory_actions, rewards, next_image_states, next_global_states, dones, state, player_id):
        #TODO: Mixed precision
        image_states = to_torch(image_states)
        global_states = to_torch(global_states)
        next_image_states = to_torch(next_image_states)
        next_global_states = to_torch(next_global_states)
        dones = to_torch(dones)
        rewards = to_torch(rewards)
        
        
        self.optim.zero_grad()

        with torch.no_grad():
            Q_next = self.model(next_image_states, next_global_states)
            Q_next_units = self._unit_mask_actions(Q_next[0], state, player_id).max(dim=1).values
            Q_next_factories = self._factories_mask_actions(Q_next[1], state, player_id).max(dim=1).values
            
        Q_target_units = rewards.unsqueeze(1).unsqueeze(1) + (self.gamma * (1 - dones)).unsqueeze(1).unsqueeze(1) * Q_next_units
        Q_target_factories = rewards.unsqueeze(1).unsqueeze(1) + (self.gamma * (1 - dones)).unsqueeze(1).unsqueeze(1) * Q_next_factories

        Q = self.model(image_states, global_states)
        Q_units = torch.gather(Q[0], dim = 1, index=unit_actions.unsqueeze(1))
        Q_factories = torch.gather(Q[1], dim = 1, index=factory_actions.unsqueeze(1))

        loss = torch.mean((Q_units - Q_target_units)**2)+torch.mean((Q_factories - Q_target_factories)**2)

        loss.backward()
        self.optim.step()

        return loss.item()