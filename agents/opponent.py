import torch
from actions.actions import UNIT_ACTION_IDXS
from agents.RL_agent import CategoricalMasked
from network.doubleconv_agent import double_conv_agent


class Opponent():
    def __init__(self, config) -> None:
        self.device = config["device"]

        self.model = double_conv_agent(config)
        self.sample_method = config["mode_or_sample"]
        assert self.sample_method in ["mode", "sample"], f"Mode or sample must be either mode or sample, found: {self.sample_method}"
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

        factory_logits, light_unit_logits, heavy_unit_logits, state_val = self.model(state["features"])


        factory_dist = CategoricalMasked(logits = factory_logits, masks = state["invalid_factory_action_mask"], device = self.device)
        light_unit_dist = CategoricalMasked(logits = light_unit_logits, masks = state["invalid_unit_action_mask"], device = self.device)
        heavy_unit_dist = CategoricalMasked(logits = heavy_unit_logits, masks = state["invalid_unit_action_mask"], device = self.device)

        if self.sample_method == "sample":
            return factory_dist.sample().squeeze().cpu().numpy(), light_unit_dist.sample().squeeze().cpu().numpy(), heavy_unit_dist.sample().squeeze().cpu().numpy(), state_val
        elif self.sample_method == "mode":
            return factory_dist.mode.squeeze().cpu().numpy(), light_unit_dist.mode.squeeze().cpu().numpy(), heavy_unit_dist.mode.squeeze().cpu().numpy(), state_val