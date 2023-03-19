from functools import partial
from lux.kit import EnvConfig
import numpy as np
import jax.numpy as jnp
from jux.torch import to_torch
from actions.old.idx_to_lux_move import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
import jax
import torch.nn.functional as F
from TD import TD
from ppo import PPO

class Agent():
    def __init__(self, env_cfg: EnvConfig, config) -> None:
        self.env_cfg: EnvConfig = env_cfg
        self.device = config["device"]
        self.map_size = config['map_size']

        self.unit_actions_per_cell = UNIT_ACTION_IDXS
        self.factory_actions_per_cell = FACTORY_ACTION_IDXS

        self.TD: TD = TD(config)

        if config["path"] is not None:
            success = self.TD.load(config["path"])
            if success:
                print("Successfully loaded model")
            else:
                print("Failed to load model")
                exit()
        self.num_envs = config["parallel_envs"]
        self.window = self.make_window(self.num_envs)


    def make_window(self, num_envs):
        window = jnp.ones((num_envs, 3, 11, 11))
        for i in range(0, 5):
            window = window.at[:, 2, i+1:10-i, i+1:10-i].set(2*i * jnp.ones((num_envs, 9-2*i, 9-2*i)))
            window = window.at[:, 1, i+1:10-i, i+1:10-i].set(i * jnp.ones((num_envs, 9-2*i, 9-2*i)))
            window = window.at[:, 0, i+1:10-i, i+1:10-i].set(-i * jnp.ones((num_envs, 9-2*i, 9-2*i)))
        return window.at[:, 1:, 5:8, 5:8].set(jnp.zeros((num_envs, 2, 3, 3)))

    @partial(jax.jit, static_argnums = (0, ))
    def early_setup(self, step: int, state, valid_spawn_mask, remainingOverageTime: int = 60):
        """board = jnp.zeros((self.num_envs, 3, 48, 48))
        board.at[:, 0, :, :].set(state.board.map.rubble/jnp.linalg.norm(state.board.map.rubble, axis = (1, 2), keepdims=True))
        board.at[:, 1, :, :].set(state.board.map.ore/jnp.linalg.norm(state.board.map.ore, axis = (1, 2), keepdims=True))
        board.at[:, 2, :, :].set(state.board.map.ice/jnp.linalg.norm(state.board.map.ice, axis = (1, 2), keepdims=True))
        
        final = jax.lax.conv_general_dilated(
            board, self.window, (1, 1), padding = "same")
        final = jnp.sum(final, axis=1)"""
        final = np.random.randint(0, 1000, size = (self.num_envs, 48, 48)) #TODO: Remove and fix convs
        #TODO: Some bug here...
        final = jnp.where(valid_spawn_mask, final, -jnp.inf)
        idx = final.reshape(final.shape[0],-1).argmax(-1)
        out = jnp.unravel_index(idx, final.shape[-2:])

        return jnp.array(out, dtype=jnp.int8).T, jnp.ones(valid_spawn_mask.shape[0], dtype=jnp.int16)*150, jnp.ones(valid_spawn_mask.shape[0], dtype=jnp.int16)*150
    
    def act(self, state, image_features, global_features, player, remainingOverageTime: int = 60):

        image_features = to_torch(image_features)
        global_features = to_torch(global_features)
        
        # Batch_size x action_space x map_size x map_size
        return list(self.TD.predict(state, image_features, global_features, player))





