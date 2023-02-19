from .idx_to_lux_move import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from math import floor
import torch
from utils.utils import load_config
from jax import numpy as jnp
from jux_wrappers.observation_wrapper import jit_create_mask_from_pos
from jux.torch import to_torch
import jax

MASK_EPS = 1e-7

config = load_config()
device = config["device"]
map_size = config["map_size"]
num_envs = config["parallel_envs"]


def calculate_move_cost(x, y, base_cost, modifier, rubble, dir):
    if dir == "right":
        dir = [1, 0]
    elif dir == "left":
        dir = [-1, 0]
    elif dir == "up":
        dir = [0, -1]
    elif dir == "down":
        dir = [0, 1]
    
    return floor(base_cost+ modifier*rubble[x+dir[0], y+dir[1]])

def can_transfer_to_tile(x, y, unit_pos, factory_pos):
    pos = [x, y]
    unit_pos = [list(elem) for elem in unit_pos] #Sometimes this is a numpy array...
    if (pos in unit_pos) or (pos in factory_pos):
        return True
    return False


#IDX to action:
#0-4 = move
#5 = dig
#6 = recharge
#7 = self_destruct

def single_unit_action_mask(unit, factory_pos, unit_pos, obs, device, player = "player_0"):
    """Calculates the action mask for one specific unit"""

    action_mask = torch.ones(UNIT_ACTION_IDXS, device=device, dtype=torch.uint8)
    #(x, y) coordinates of unit
    x, y = unit["pos"]
    #Power remaining for unit, i.e charge
    power = unit["power"] - (1 if unit["unit_type"] == "LIGHT" else 10) #Subtracting the cost of adding to action queue
    #48x48 map of how much rubble is in which square
    rubble = obs["board"]["rubble"]
    #How much digging costs for this unit
    dig_cost = 5 if unit["unit_type"] == "LIGHT" else 60
    #The base cost of moving for this unit
    move_cost_base = 1 if unit["unit_type"] == "LIGHT" else 20
    #The modifier for how much extra it costs to move over each rubble
    rubble_move_modifier = 0.05 if unit["unit_type"] == "LIGHT" else 1

    #Move
    # (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    # (0, 0) is top left #TODO: Is this true?
    #TODO: Should check if there is a factory there
    if x == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "left"):
        action_mask[4] = MASK_EPS
    if x == 47 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "right"):
        action_mask[2] = MASK_EPS
    if y == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "up"):
        action_mask[1] = MASK_EPS
    if y == 47 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "down"):
        action_mask[3] = MASK_EPS

    #Dig
    if unit["power"] < dig_cost or list(unit["pos"]) in factory_pos:
        action_mask[5] = MASK_EPS

    #Recharge, max is 150 for LIGHT and 3000 for heavy per config
    #TODO: This should be based on how much is recharged each turn
    if  power == (150 if unit["unit_type"] else 3000):
        action_mask[6] = MASK_EPS
    
    #Self destruct, requires 10 power
    #NOTE: Self destruct is illegal for now
    if True: #power < (10 if unit["unit_type"] == "LIGHT" else 100):
        action_mask[7] = MASK_EPS

    #TODO: Only center
    #Transport ice
    if unit["cargo"]["ice"] == 0 or not can_transfer_to_tile(x, y, unit_pos, factory_pos):
        action_mask[8] = MASK_EPS

    #TODO: Only center
    #Transport ore
    if unit["cargo"]["ore"] == 0 or not can_transfer_to_tile(x, y, unit_pos, factory_pos):
        action_mask[9] = MASK_EPS

    return action_mask

def calculate_water_cost(x, y, obs):
    #TODO Implement this
    #Should be ceil(number of connected and new lichen tiles / 10)
    return 10

#IDX to action
#0 -> LIGHT 
#1 -> HEAVY
#2 -> LICHEN
#3 -> NOTHING
def single_factory_action_mask(factory, obs, device):
    action_mask = torch.ones(FACTORY_ACTION_IDXS, device=device, dtype=torch.uint8)
    
    metal = factory["cargo"]["metal"]
    water = factory["cargo"]["water"]
    power = factory["power"]

    if metal < 10 or power < 50:
        action_mask[0] = MASK_EPS

    if metal < 100 or power < 500:
        action_mask[1] = MASK_EPS
    
    if water < calculate_water_cost(*factory["pos"], obs):
        action_mask[2] = MASK_EPS

    return action_mask


def unit_action_mask(obs, device, player = "player_0"):
    #NOTE: Needs to take in a player for the factory stuff?
    obs = obs[player]
    action_mask = torch.ones((map_size, map_size, UNIT_ACTION_IDXS), device=device, dtype=torch.uint8)

    #Get factory position
    factories = obs["factories"][player]
    factory_pos = [factory["pos"] for _, factory in factories.items()]
    
    #Get unit position
    units = obs["units"][player]
    unit_pos = [unit["pos"] for _, unit in units.items()]

    for unit in units.values():
        x, y = unit["pos"]
        action_mask[x, y] = single_unit_action_mask(unit, factory_pos, unit_pos, obs, device, player)

    return action_mask


def factory_action_mask(obs, device, player = "player_0"):
    #note: Needs to take in a player for the factory stuff?
    action_mask = torch.ones((map_size, map_size, FACTORY_ACTION_IDXS), device=device, dtype=torch.uint8)
    obs = obs[player]
    factories = obs["factories"][player]
    for factory in factories.values():
        x, y = factory["pos"]
        mask = single_factory_action_mask(factory, obs, device)
        action_mask[x, y] = mask

    return action_mask


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
    return ~mask.permute(0, 2, 3, 1)