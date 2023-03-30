from actions.actions import outputs_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from torch.distributions import Categorical
import torch
from network.ActorCritic import ActorCritic
import numpy as np

class Agent():
    def __init__(self, player: str, config) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.device = config["device"]

        self.sample_method = config["mode_or_sample"]
        assert self.sample_method in ["mode", "sample"], f"Mode or sample must be either mode or sample, found: {self.sample_method}"

        self.unit_actions_per_cell = UNIT_ACTION_IDXS
        self.factory_actions_per_cell = FACTORY_ACTION_IDXS


        self.model = ActorCritic(UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, config)

        if config["path"] is not None:
            self.model.load_state_dict(torch.load(config["path"], map_location=lambda storage, loc: storage))
            print("Successfully loaded model, this is not a fresh run")
        else:
            print("This is a fresh run! Good luck")


    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def get_value(self, image_features, global_features):
        return self.model.forward_critic(image_features, global_features)

    def get_action_and_value(self, image_features, global_features, unit_action_mask, factory_action_mask, action_unit=None, action_factory=None, testing = False):
        #NOTE: Assumes first channel is unit mask for our agent
        #NOTE: Assumes second channel is factory mask for our agent

        action_probs_unit, action_probs_factories = self.model.forward_actor(image_features, global_features)

        assert action_probs_unit.shape == unit_action_mask.shape, f"Prob shape: {action_probs_unit.shape}, Mask shape: {unit_action_mask.shape}"
        assert action_probs_factories.shape == factory_action_mask.shape, f"Prob shape: {action_probs_factories.shape}, Mask shape: {factory_action_mask.shape}"

        action_probs_unit = torch.where(unit_action_mask, action_probs_unit, -1e8)

        action_probs_factories = torch.where(factory_action_mask, action_probs_factories, -1e8)

        unit_dist, factory_dist = Categorical(logits = action_probs_unit), Categorical(logits = action_probs_factories)

        if action_unit is None:
            if self.sample_method == "mode":
                action_unit = unit_dist.mode
                action_factory = factory_dist.mode
            else:
                action_unit = unit_dist.sample()
                action_factory = factory_dist.sample()

        action_logprob_unit = unit_dist.log_prob(action_unit)*(image_features[:, 0] == 1)
        action_logprob_factory = factory_dist.log_prob(action_factory)*(image_features[:, 1] == 1)

        state_val = self.model.forward_critic(image_features, global_features)

        return action_unit, action_factory, (action_logprob_unit).sum((1, 2)), (action_logprob_factory).sum((1, 2)), unit_dist.entropy(), factory_dist.entropy(), state_val
    
    def get_action(self, image_features, global_features, unit_action_mask, factory_action_mask, testing = False):
        if len(image_features.shape) == 3:
            image_features = image_features.unsqueeze(0)
            global_features = global_features.unsqueeze(0)
            unit_action_mask = unit_action_mask.unsqueeze(0)
            factory_action_mask = factory_action_mask.unsqueeze(0)

        unit_action, factory_action, _, _, _, _, _ = self.get_action_and_value(image_features, global_features, unit_action_mask, factory_action_mask, testing = False)
        dim = 0 if unit_action.ndim == 3 else 1
        return np.stack((unit_action.squeeze().cpu().numpy(), factory_action.squeeze().cpu().numpy()), axis = dim)

