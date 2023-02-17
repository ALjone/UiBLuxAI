from email.mime import image
from lux.kit import obs_to_game_state, EnvConfig
from lux.utils import my_turn_to_place_factory
import numpy as np
import jax.numpy as jnp
from jax import scipy as jsp
from jux.torch import from_torch
from actions.idx_to_lux_move import outputs_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, unit_id_to_action_idx
from ppo import PPO
import jax
import torch.nn.functional as F
from jux_wrappers.observation_wrapper import _image_features, observation
from TD import TD

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig, config) -> None:
        self.player = player
        self.opp_player = 1 if self.player == 0 else 1
        self.env_cfg: EnvConfig = env_cfg
        self.device = config["device"]
        self.map_size = config['map_size']

        self.unit_actions_per_cell = UNIT_ACTION_IDXS
        self.factory_actions_per_cell = FACTORY_ACTION_IDXS

        self.TD = TD()

        if config["path"] is not None:
            self.PPO.load(config["path"])
            print("Successfully loaded model")
        num_envs = config["parallel_envs"]
        self.window = self.make_window(num_envs)
    def make_window(self, num_envs):
        window = jnp.ones((num_envs, 3, 11, 11))
        for i in range(0, 5):
            window = window.at[:, 2, i+1:10-i, i+1:10-i].set(2*i * jnp.ones((num_envs, 9-2*i, 9-2*i)))
            window = window.at[:, 1, i+1:10-i, i+1:10-i].set(i * jnp.ones((num_envs, 9-2*i, 9-2*i)))
            window = window.at[:, 0, i+1:10-i, i+1:10-i].set(-i*jnp.ones((num_envs, 9-2*i, 9-2*i)))
        return window.at[:, 1:, 5:8, 5:8].set(jnp.zeros((num_envs, 2, 3, 3)))

    def early_setup(self, step: int, state, valid_spawn_mask, remainingOverageTime: int = 60):
        num_envs = 50
        map = jnp.zeros((num_envs, 3, 48, 48))
        map.at[:, 0, :, :].set(state.board.map.rubble/jnp.linalg.norm(state.board.map.rubble, axis = (1, 2), keepdims=True))
        map.at[:, 1, :, :].set(state.board.map.ore/jnp.linalg.norm(state.board.map.ore, axis = (1, 2), keepdims=True))
        map.at[:, 2, :, :].set(state.board.map.ice/jnp.linalg.norm(state.board.map.ice, axis = (1, 2), keepdims=True))
        
        final = jax.lax.conv_general_dilated(
            map, self.window, (1, 1), padding = "same")
        final = jnp.sum(final, axis=1)
        final = jnp.where(valid_spawn_mask, final, -jnp.inf)


        idx = final.reshape(final.shape[0],-1).argmax(-1)
        out = jnp.unravel_index(idx, final.shape[-2:])

        return jnp.array(out).T, jnp.ones(valid_spawn_mask.shape[0])*150, jnp.ones(valid_spawn_mask.shape[0])*150
    
    def act(self, state, image_features, global_features, remainingOverageTime: int = 60):
        
        # Batch_size x action_space x map_size x map_size
        pred_units, pred_factories = self.TD.predict(state, image_features, global_features)

        # TODO: new model_output to action_format functions
        jux_actions = actions_to_jux(pred_units, pred_factories)
        return jux_actions




