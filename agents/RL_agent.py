from torch.distributions import Categorical
import torch
from network.doubleconv_agent import double_conv_agent
from actions.actions import UNIT_ACTION_IDXS

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], device = torch.device("cpu")):
        self.masks = masks
        self.device = device
        self.masks = masks.type(torch.BoolTensor).to(device)
        logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)


class Agent():
    def __init__(self, config) -> None:
        self.device = config["device"]

        self.mean_entropy = config["mean_entropy"]
        self.sample_method = config["mode_or_sample"]
        assert self.sample_method in ["mode", "sample"], f"Mode or sample must be either mode or sample, found: {self.sample_method}"


        print("Running with sampling method:", self.sample_method)


        self.model = double_conv_agent(config)
        print("Model has:", self.model.count_parameters(), "parameters")

        if config["path"] is not None:
            state_dict = torch.load(config["path"])
            self.model.load_state_dict(state_dict)
            print("Successfully loaded model, this is not a fresh run")
    
        else:
            print("This is a fresh run! Good luck")


    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def get_value(self, image_features):
        _, _, _,  state_val = self.model(image_features)
        return state_val
    
    def get_action_probs(self, state):
        factory_logits, light_unit_logits, heavy_unit_logits, _ = self.model(state["features"])
        return factory_logits.squeeze(), light_unit_logits.squeeze(), heavy_unit_logits.squeeze()

    def get_action_and_value(self, state, factory_action = None, light_unit_action = None, heavy_unit_action = None, return_dist = False):
        #NOTE: Assumes first channel is unit mask for our agent

        factory_logits, light_unit_logits, heavy_unit_logits, state_val  = self.model(state["features"])


        if len(state["invalid_unit_action_mask"].shape) == 3:
            state["invalid_unit_action_mask"] = state["invalid_unit_action_mask"].unsqueeze(0)
            state["invalid_factory_action_mask"] = state["invalid_factory_action_mask"].unsqueeze(0)
            state["unit_mask"] = state["unit_mask"].unsqueeze(0)
            state["factory_mask"] = state["factory_mask"].unsqueeze(0)

        if return_dist:
            factory_mask = state["factory_mask"].unsqueeze(-1).repeat_interleave(4, -1)
            unit_mask = state["unit_mask"].unsqueeze(-1).repeat_interleave(11, -1)
            
            factory_dist_for_KL = torch.nn.LogSoftmax(dim = 3)(factory_logits)*factory_mask

            light_unit_dist_for_KL = torch.nn.LogSoftmax(dim = 3)(light_unit_logits)*unit_mask

            heavy_unit_dist_for_KL = torch.nn.LogSoftmax(dim = 3)(heavy_unit_logits)*unit_mask

        else:
            factory_dist_for_KL = None
            light_unit_dist_for_KL = None
            heavy_unit_dist_for_KL = None

        factory_dist = CategoricalMasked(logits = factory_logits, masks = state["invalid_factory_action_mask"], device = self.device)
        
        light_unit_dist = CategoricalMasked(logits = light_unit_logits, masks = state["invalid_unit_action_mask"], device = self.device)
        heavy_unit_dist = CategoricalMasked(logits = heavy_unit_logits, masks = state["invalid_unit_action_mask"], device = self.device)
        
        if light_unit_action is None:
            if self.sample_method == "mode":
                factory_action = factory_dist.mode
                light_unit_action = light_unit_dist.mode
                heavy_unit_action = heavy_unit_dist.mode
            elif self.sample_method == "sample":
                factory_action = factory_dist.sample()
                light_unit_action = light_unit_dist.sample()
                heavy_unit_action = heavy_unit_dist.sample()

        factory_action_logprob = (factory_dist.log_prob(factory_action) * state["factory_mask"]).sum((1, 2))
        light_unit_action_logprob = (light_unit_dist.log_prob(light_unit_action) * state["unit_mask"]).sum((1, 2))
        heavy_unit_action_logprob = (heavy_unit_dist.log_prob(heavy_unit_action) * state["unit_mask"]).sum((1, 2))

        factory_entropy = (factory_dist.entropy() * state["factory_mask"]).sum((1, 2))
        light_unit_entropy = (light_unit_dist.entropy() * state["unit_mask"]).sum((1, 2))
        heavy_unit_entropy = (heavy_unit_dist.entropy() * state["unit_mask"]).sum((1, 2))

        return  factory_action, factory_action_logprob, factory_entropy, factory_dist_for_KL, \
                light_unit_action, light_unit_action_logprob, light_unit_entropy, light_unit_dist_for_KL, \
                heavy_unit_action,  heavy_unit_action_logprob,  heavy_unit_entropy, heavy_unit_dist_for_KL, \
                state_val


    def get_dist_and_value(self, features: torch.Tensor, unit_mask:torch.Tensor, factory_mask: torch.Tensor):
        factory_logits, light_unit_logits, heavy_unit_logits, state_val = self.model(features)

        factory_mask = factory_mask.unsqueeze(-1).repeat_interleave(4, -1)
        unit_mask = unit_mask.unsqueeze(-1).repeat_interleave(UNIT_ACTION_IDXS, -1)

        factory_dist = torch.nn.LogSoftmax(dim = 3)(factory_logits)*factory_mask
        light_unit_dist = torch.nn.LogSoftmax(dim = 3)(light_unit_logits)*unit_mask
        heavy_unit_dist = torch.nn.LogSoftmax(dim = 3)(heavy_unit_logits)*unit_mask


        return factory_dist, light_unit_dist, heavy_unit_dist, state_val