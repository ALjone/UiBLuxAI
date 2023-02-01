from .move_utils import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from math import floor
import torch


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

#IDX to action:
#0-4 = move
#5 = dig
#6 = recharge
#7 = self_destruct

def single_unit_action_mask(unit, obs, device, player = "player_0"):
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
    if x == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "left"):
        action_mask[4] = 0
    if x == 47 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "right"):
        action_mask[2] = 0
    if y == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "up"):
        action_mask[1] = 0
    if y == 47 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "down"):
        action_mask[3] = 0

    factories = obs["factories"][player]
    factory_pos = [factory["pos"] for _, factory in factories.items()]

    #Dig
    if unit["power"] < dig_cost or list(unit["pos"]) in factory_pos:
        action_mask[5] = 0

    #Recharge, max is 150 for LIGHT and 3000 for heavy per config
    #TODO: This should be based on how much is recharged each turn
    if  power == (150 if unit["unit_type"] else 3000):
        action_mask[6] = 0
    
    #Self destruct, requires 10 power
    if power < (10 if unit["unit_type"] == "LIGHT" else 100):
        action_mask[7] = 0 

    #Transport ice
    if unit["cargo"]["ice"] == 0:
        action_mask[8] = 0

    #Transport ore
    if unit["cargo"]["ore"] == 0:
        action_mask[9] = 0

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
        action_mask[0] = 0

    if metal < 100 or power < 500:
        action_mask[1] = 0
    
    if water < calculate_water_cost(*factory["pos"], obs):
        action_mask[2] = 0

    return action_mask


def unit_action_mask(obs, device, player = "player_0"):
    #NOTE: Needs to take in a player for the factory stuff?
    obs = obs[player]
    action_mask = torch.ones((48, 48, UNIT_ACTION_IDXS), device=device, dtype=torch.uint8)
    units = obs["units"][player]
    for unit in units.values():
        x, y = unit["pos"]
        action_mask[x, y] = single_unit_action_mask(unit, obs, player)

    return action_mask


def factory_action_mask(obs, device, player = "player_0"):
    #note: Needs to take in a player for the factory stuff?
    action_mask = torch.ones((48, 48, FACTORY_ACTION_IDXS), device=device, dtype=torch.uint8)
    obs = obs[player]
    factories = obs["units"][player]
    for factory in factories.values():
        x, y = factory["pos"]
        mask = single_factory_action_mask(factory, obs, device)
        print(mask)
        action_mask[x, y] = #single_factory_action_mask(factory, obs, device)

    return action_mask