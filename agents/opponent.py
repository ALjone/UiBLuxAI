import torch
from actions.actions import UNIT_ACTION_IDXS
from agents.RL_agent import CategoricalMasked
from network.ActorCritic import ActorCritic


class Opponent():
    def __init__(self, config) -> None:
        self.device = config["device"]

        self.sample_method = "mode"
        self.model = ActorCritic(UNIT_ACTION_IDXS, config)
        #del self.model.critic

    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def load(self, path):
        if isinstance(path, str):
            self.model.load_state_dict(torch.load(path))
        elif isinstance(path, dict):
            self.model.load_state_dict(path)
        else:  
            raise ValueError("Path should be string or dict, found:", type(path))
        self.model.eval()

    def get_action(self, state):
        #NOTE: Assumes first channel is unit mask for our agent
        
        for key, val in state.items():
            state[key] = torch.tensor(val).to(self.device)

        unit_action_probs, factory_action_probs = self.model.forward_actor(state["features"])


        unit_dist = CategoricalMasked(logits = unit_action_probs, masks = state["invalid_unit_action_mask"], device = self.device)
        factory_dist = CategoricalMasked(logits = factory_action_probs, masks = state["invalid_factory_action_mask"], device = self.device)

        return unit_dist.mode.squeeze().cpu().numpy(), factory_dist.mode.squeeze().cpu().numpy()