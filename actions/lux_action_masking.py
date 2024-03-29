from .old.idx_to_lux_move import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from math import floor
import torch
from utils.utils import load_config

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
    #TODO: Not implemented
    action_mask = torch.zeros(UNIT_ACTION_IDXS, device=device, dtype=torch.uint8)
    action_mask[[(0, 1, 2, 3, 4, 5, 6, 7, 8, UNIT_ACTION_IDXS-2)]] = 1
    return action_mask

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
    action_mask = torch.ones((UNIT_ACTION_IDXS, map_size, map_size), device=device, dtype=torch.uint8)

    #Get factory position
    factories = obs["factories"][player]
    factory_pos = [factory["pos"] for _, factory in factories.items()]
    
    #Get unit position
    units = obs["units"][player]
    unit_pos = [unit["pos"] for _, unit in units.items()]

    for unit in units.values():
        x, y = unit["pos"]
        action_mask[:, x, y] = single_unit_action_mask(unit, factory_pos, unit_pos, obs, device, player)

    return action_mask.unsqueeze(0).to(torch.bool)


def factory_action_mask(obs, device, player = "player_0"):
    #note: Needs to take in a player for the factory stuff?
    action_mask = torch.ones((FACTORY_ACTION_IDXS, map_size, map_size), device=device, dtype=torch.uint8)
    obs = obs[player]
    factories = obs["factories"][player]
    for factory in factories.values():
        x, y = factory["pos"]
        mask = single_factory_action_mask(factory, obs, device)
        action_mask[:, x, y] = mask

    return action_mask.unsqueeze(0).to(torch.bool)
