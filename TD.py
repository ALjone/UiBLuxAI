import torch
from network.actor import Actor
from actions.idx_to_lux_move import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from actions.action_masking import batched_action_mask, unit_action_mask
class TD:
    def __init__(self, config) -> None:
        self.device = config["device"]
        self.model  = Actor(36, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, config["actor_n_blocks"], config["actor_n_blocks_after_split"], config["actor_intermediate_channels"], config).to(self.device)

        self.gamma = config["gamma"]
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["regularization"])

        self.action_space = UNIT_ACTION_IDXS

        self.batch_size = config["batch_size"]

    def _unit_mask_actions(self, outputs, state, player_id):
        #1 - player_id to get opponent id
        mask = batched_action_mask(state, player_id)
        outputs[mask] = -torch.inf
        return outputs

    def _factories_mask_actions(self, outputs, state):
        return outputs
        raise NotImplementedError()
    
    def predict(self, state, image_features, global_features, player_id):
            with torch.no_grad():
                self.model.eval()
                pred_units, pred_factories = self.model(image_features, global_features)
                unit_actions = self._unit_mask_actions(pred_units, state, player_id)
                factories_actions = self._factories_mask_actions(pred_factories, state)

                return torch.argmax(unit_actions, dim = 1), torch.argmax(factories_actions, dim = 1) #NOTE: Assumes channel first


    def train(self, states, actions, rewards, next_states, dones):

        self.optim.zero_grad()


        with torch.no_grad():
            Q_next = self.target_model(next_states).max(dim=1).values
        Q_target = rewards + self.gamma * (1 - dones) * Q_next
        Q = self.model(states)[torch.arange(len(actions)), actions.to(torch.long).flatten()]

        loss = torch.mean((Q - Q_target)**2)

        loss.backward()
        self.optim.step()

        return loss.item()