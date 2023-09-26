from actions.actions import UNIT_ACTION_IDXS
from torch.distributions import Categorical
import torch
from network.ActorCritic import ActorCritic


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

        self.sample_method = config["mode_or_sample"]
        assert self.sample_method in ["mode", "sample"], f"Mode or sample must be either mode or sample, found: {self.sample_method}"


        print("Running with sampling method:", self.sample_method)


        self.model = ActorCritic(UNIT_ACTION_IDXS, config)
        print("Actor has:", self.model.count_actor_parameters(), "parameters")
        print("Critic has:", self.model.count_critic_parameters(), "parameters")

        if config["path"] is not None:
            self.model.load_state_dict(torch.load(config["path"]))
            print("Successfully loaded model, this is not a fresh run")
        else:
            print("This is a fresh run! Good luck")


    def save(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def get_value(self, image_features):
        state_val = self.model.forward_critic(image_features)
        return state_val
    
    def get_action_probs(self, state):
        unit_action_probs, factory_action_probs = self.model.forward_actor(state["features"])
        return torch.nn.functional.softmax(unit_action_probs, -1).squeeze(), torch.nn.functional.softmax(factory_action_probs, -1).squeeze()

    def get_action_and_value(self, state, unit_action = None, factory_action = None):
        #NOTE: Assumes first channel is unit mask for our agent

        unit_action_probs, factory_action_probs  = self.model.forward_actor(state["features"])
        state_val = self.model.forward_critic(state["features"])

        assert unit_action_probs.shape == state["invalid_unit_action_mask"].shape
        unit_dist = CategoricalMasked(logits = unit_action_probs, masks = state["invalid_unit_action_mask"], device = self.device)
        # Categorical(logits = torch.where(state["invalid_unit_action_mask"].to(torch.bool).to(self.device), unit_action_probs, torch.tensor(-1e+8)), device = self.device)
        factory_dist = CategoricalMasked(logits = factory_action_probs, masks = state["invalid_factory_action_mask"], device = self.device)

        if unit_action is None:
            if self.sample_method == "mode":
                unit_action = unit_dist.mode
                factory_action = factory_dist.mode
            else:
                unit_action = unit_dist.sample()
                factory_action = factory_dist.sample()

        unit_action_logprob = (unit_dist.log_prob(unit_action) * state["unit_mask"]).sum((1, 2))
        factory_action_logprob = (factory_dist.log_prob(factory_action) * state["factory_mask"]).sum((1, 2))

        unit_entropy = (unit_dist.entropy() * state["unit_mask"]).sum((1, 2))
        factory_entropy = (factory_dist.entropy() * state["factory_mask"]).sum((1, 2))

        return unit_action, unit_action_logprob, unit_entropy, factory_action, factory_action_logprob, factory_entropy, state_val

    
    def get_action(self, obs):
        if len(obs["features"].shape) == 3:
            for key, item in obs.items():
                obs[key] = torch.tensor(item).unsqueeze(0)

        unit_action, _, _, factory_action, _, _, _ = self.get_action_and_value(obs)

        return unit_action.squeeze().cpu().numpy(), factory_action.squeeze().cpu().numpy()

    def get_dist_and_value(self, features: torch.Tensor, unit_mask:torch.Tensor, factory_mask: torch.Tensor):
        unit_action_probs, factory_action_probs = self.model.forward_actor(features)
        state_val = self.model.forward_critic(features)

        unit_mask = unit_mask.unsqueeze(-1).repeat_interleave(11, -1)
        factory_mask = factory_mask.unsqueeze(-1).repeat_interleave(4, -1)

        unit_dist = torch.nn.LogSoftmax(dim = 3)(unit_action_probs)*unit_mask
        #unit_dist = unit_dist[unit_mask.to(torch.bool).flatten(1)]

        factory_dist = torch.nn.LogSoftmax(dim = 3)(factory_action_probs)*factory_mask
        #factory_dist = factory_dist[factory_mask.to(torch.bool).flatten(1)]

        return unit_dist, factory_dist, state_val