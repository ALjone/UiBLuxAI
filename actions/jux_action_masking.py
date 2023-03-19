from .old.idx_to_lux_move import UNIT_ACTION_IDXS
import torch
from jax import numpy as jnp
from jux_wrappers.observation_wrapper import jit_create_mask_from_pos
from jux.torch import to_torch
import jax
from utils.utils import load_config

config = load_config()
device = config["device"]
map_size = config["map_size"]
num_envs = config["parallel_envs"]

def get_factories(state, player_idx):
    factory_x = state.factories.pos.x[:, player_idx]
    factory_y = state.factories.pos.y[:, player_idx]

    friendly_factory_mask = jit_create_mask_from_pos(jnp.zeros((num_envs, map_size, map_size), dtype=jnp.bool_), factory_x, factory_y, True)

    friendly_factory_mask = friendly_factory_mask & jnp.roll(friendly_factory_mask, 1, axis=1)
    friendly_factory_mask = friendly_factory_mask & jnp.roll(friendly_factory_mask, -1, axis=1)
    friendly_factory_mask = friendly_factory_mask & jnp.roll(friendly_factory_mask, 1, axis=2)
    friendly_factory_mask = friendly_factory_mask & jnp.roll(friendly_factory_mask, -1, axis=2)

    return friendly_factory_mask, state.board.factory_occupancy_map - friendly_factory_mask

jit_get_factories = jax.jit(get_factories)
#NORTH, EAST, SOUTH, WEST
def batched_action_mask(state, player_idx):
    mask = torch.ones((num_envs, UNIT_ACTION_IDXS, map_size, map_size), dtype = torch.bool).to(device)
    # TODO: Check the dimensions, think x,y and i,j is being mixed up

    #FIX THIS
    friendly_factory_pos, enemy_factory_pos = jit_get_factories(state, player_idx)
    friendly_factory_pos = to_torch(friendly_factory_pos).to(torch.bool)
    enemy_factory_pos = to_torch(enemy_factory_pos).to(torch.bool)

    #TODO: Assumes matrix indexing, plz check
    # Also assumes x valiue before y value
    # y - 1
    #TODO: Also convert to jnp and jit
    #TODO: This was rewritten without checking how it works, so this needs to be
    #double checked
    factory_north = torch.roll(enemy_factory_pos, 1, dims=2)
    factory_north[:, :, 0] = True

    factory_east = torch.roll(enemy_factory_pos, -1, dims=1)
    factory_east[:, -1, :] = True

    factory_south = torch.roll(enemy_factory_pos, -1, dims=2)
    factory_south[:, :, -1] = True

    factory_west = torch.roll(enemy_factory_pos, 1, dims=1)
    factory_west[:, 0, :] = True

    mask[:, 1] *= ~factory_north
    mask[:, 2] *= ~factory_east
    mask[:, 3] *= ~factory_south
    mask[:, 4] *= ~factory_west


    # Transfer mask
    #TODO: Fix
    #mask[:, [5, 9, 13, 17, 18, 19, 20]] *= ~factory_north
    #mask[:, [6, 10, 14, 21, 22, 23, 24]] *= ~factory_east
    #mask[:, [7, 11, 15, 25, 26, 27, 28]] *= ~factory_south
    #mask[:, [8, 12, 16, 29, 30, 31, 32]] *= ~factory_west

    # Pickup and dig
    #mask[:, 33:38] *= ~friendly_factory_pos

    #Flip to return illegal moves TODO: Fix this at the top so we don't have to permute here
    return (~mask)
