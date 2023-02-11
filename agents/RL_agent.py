from lux.kit import obs_to_game_state, EnvConfig
from lux.utils import my_turn_to_place_factory
import numpy as np
import jax.numpy as jnp
from jax import scipy as jsp
from jux.torch import from_torch
from actions.idx_to_lux_move import outputs_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, unit_id_to_action_idx
from ppo import PPO
import torch
import torch.nn.functional as F
class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig, config) -> None:
        self.player = player
        self.opp_player = 1 if self.player == 0 else 1
        self.env_cfg: EnvConfig = env_cfg
        self.device = config["device"]

        self.unit_actions_per_cell = UNIT_ACTION_IDXS
        self.factory_actions_per_cell = FACTORY_ACTION_IDXS


        self.PPO = PPO(self.unit_actions_per_cell, self.factory_actions_per_cell, config)

        if config["path"] is not None:
            self.PPO.load(config["path"])
            print("Successfully loaded model")

    def early_setup(self, step: int, state, valid_spawn_mask, remainingOverageTime: int = 60):
        num_envs = valid_spawn_mask.shape[0]
        if step == 0:
            raise ValueError("You're supposed to handle bidding outside of this...")
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            map = torch.zeros((num_envs, 3, valid_spawn_mask.shape[1], valid_spawn_mask.shape[2]), dtype=torch.float32)
            map[:, 0, :, :] = (state.board.map.rubble/torch.linalg.norm(state.board.map.rubble.to(torch.float32), axis = (1, 2), keepdims=True))
            map[:, 1, :, :] = (state.board.map.ore/torch.linalg.norm(state.board.map.ore.to(torch.float32), axis = (1, 2), keepdims=True))
            map[:, 2, :, :] = (state.board.map.ice/torch.linalg.norm(state.board.map.ice.to(torch.float32), axis = (1, 2), keepdims=True))

            window = torch.ones((num_envs, 3, 11, 11))
            for i in range(0, 5):
                window[:, 2, i+1:10-i, i+1:10-i] = (2*i * torch.ones((num_envs, 9-2*i, 9-2*i)))
                window[:, 1, i+1:10-i, i+1:10-i] = (i * torch.ones((num_envs, 9-2*i, 9-2*i)))
                window[:, 0, i+1:10-i, i+1:10-i] = (-i*torch.ones((num_envs, 9-2*i, 9-2*i)))
            window[:, 1:, 5:8, 5:8] = (torch.zeros((num_envs, 2, 3, 3)))

            final = F.conv2d(
                map, window, padding="same")
            final = torch.sum(final, axis=1)
            final[valid_spawn_mask == False] = -torch.inf


            idx = final.reshape(final.shape[0],-1).argmax(-1)
            out = jnp.unravel_index(from_torch(idx), final.shape[-2:])

            return jnp.array(out).T, jnp.ones(valid_spawn_mask.shape[0])*150, jnp.ones(valid_spawn_mask.shape[0])*150
    
    def act(self, state, remainingOverageTime: int = 60):
        features = state[0][self.player]
        obs = state[1]
        units = obs[self.player]["units"][self.player].values()
        factories = obs[self.player]["factories"][self.player].values()

        image_features = features["image_features"].to(self.device)
        #action_queue_type = torch.zeros((UNIT_ACTION_IDXS-1, 48, 48), device = self.device) #-1 because of the do nothing action



        #First in cat is first in output, so unit/factory mask is kept!!
        #image_features = torch.cat((image_features, action_queue_type), dim=0)

        global_features = features["global_features"].to(self.device)

        unit_output, factory_output = self.PPO.select_action(image_features, global_features, obs)

        #action_idx_dict = unit_id_to_action_idx(units, unit_output)

        #for unit_id, action in action_idx_dict.items():
        #    self.action_queue[unit_id] = action



        action = outputs_to_actions(unit_output.detach().cpu(), factory_output.detach().cpu(), units, factories, obs, factory_map = image_features[1].detach().cpu().numpy()) #Second channel is always factory_map

        return action
