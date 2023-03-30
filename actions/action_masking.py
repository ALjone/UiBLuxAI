from actions.action_mask_utils import calculate_move_cost, can_transfer_to_tile
from actions.actions import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from actions.single_move_actions import find_dir_to_closest
import torch
from utils.utils import load_config

MASK_EPS = False
MIN_TRANSFER_AMOUNT = 2

config = load_config()
device = config["device"]

MOVES = [   "No move", 
            "Move North",
            "Move East", 
            "Move South", 
            "Move West",
            "Move closest ice",
            "Move closest ore",
            "Move closest rubble",
            #Move to closest enemy lichen
            "Move closest factory",
            "Move closest enemy unit",
            "Transfer max to closest factory",
            "Dig",
            "Pick up power",
            #"Transport to closest unit",
            ]





def single_unit_action_mask(unit, obs, state, device):
    """Calculates the action mask for one specific unit"""
    action_mask = torch.ones(UNIT_ACTION_IDXS, device=device, dtype=torch.bool)
    factory_occupancy_mask = state[4]
    enemy_factory_occupancy_mask = state[5]

    #TODO: Look reaaaaaaally close if there are any bugs here

    #(x, y) coordinates of unit
    x, y = unit["pos"]
    #Power remaining for unit, i.e charge
    #TODO: Uncomment to add cost of action queue
    power = unit["power"] #- (1 if unit["unit_type"] == "LIGHT" else 10) #Subtracting the cost of adding to action queue
    #48x48 map of how much rubble is in which square
    rubble = obs["board"]["rubble"]
    ice = obs["board"]["ice"]
    ore = obs["board"]["ore"]
    #How much digging costs for this unit
    dig_cost = 5 if unit["unit_type"] == "LIGHT" else 60
    #The base cost of moving for this unit
    move_cost_base = 1 if unit["unit_type"] == "LIGHT" else 20
    #The modifier for how much extra it costs to move over each rubble
    rubble_move_modifier = 0.05 if unit["unit_type"] == "LIGHT" else 1
    
    #TODO: Should check if there is an enemy factory there

    #NOTE: Move
    # (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    # (0, 0) is top left #TODO: Is this true?
    if y == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "north", enemy_factory_occupancy_mask):
        action_mask[1] = MASK_EPS
    if x == 47 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "east", enemy_factory_occupancy_mask):
        action_mask[2] = MASK_EPS
    if y == 47 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "south", enemy_factory_occupancy_mask):
        action_mask[3] = MASK_EPS
    if x == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "west", enemy_factory_occupancy_mask):
        action_mask[4] = MASK_EPS


    #NOTE: Move closest ice
    dir = find_dir_to_closest(ice, x, y, rubble)
    if power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, dir, enemy_factory_occupancy_mask):
        action_mask[5] = MASK_EPS
    
    #NOTE: Move closest ore
    dir = find_dir_to_closest(ore, x, y, rubble)
    if power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, dir, enemy_factory_occupancy_mask):
        action_mask[6] = MASK_EPS

    #NOTE: Move closest rubble
    dir = find_dir_to_closest(rubble, x, y, rubble)
    if power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, dir, enemy_factory_occupancy_mask):
        action_mask[7] = MASK_EPS

    #NOTE: Move closest friendly factory. Friendly factories are always state[1]
    factory_dir = find_dir_to_closest(state[1], x, y, rubble)
    if power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, factory_dir, enemy_factory_occupancy_mask):
        action_mask[8] = MASK_EPS


    #NOTE: Move closest enemy unit. Enemy units are always state[2]
    if torch.sum(state[2]) >= 1:
        print("Sum is over?????????")
        dir = find_dir_to_closest(state[2], x, y, rubble)
        if power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, dir, enemy_factory_occupancy_mask):
            action_mask[9] = MASK_EPS
    else:
        action_mask[9] = MASK_EPS

    #NOTE: Transfer max to closest factory
    #NOTE: We reuse dir here, because last dir is from factory
    if not can_transfer_to_tile(x, y, factory_occupancy_mask, factory_dir) or (unit["cargo"]["ice"] < MIN_TRANSFER_AMOUNT and unit["cargo"]["ore"] < MIN_TRANSFER_AMOUNT and unit["cargo"]["water"] < MIN_TRANSFER_AMOUNT and unit["cargo"]["metal"] < MIN_TRANSFER_AMOUNT):
        action_mask[10] = MASK_EPS

    #Dig
    if unit["power"] < dig_cost or factory_occupancy_mask[x, y] == 1:
        action_mask[11] = MASK_EPS

    #Power
    if unit["power"] < MIN_TRANSFER_AMOUNT or not can_transfer_to_tile(x, y, factory_occupancy_mask, "center"):
        action_mask[12] = MASK_EPS


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
    action_mask = torch.ones(FACTORY_ACTION_IDXS, device=device, dtype=torch.bool)
    
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


def unit_action_mask(obs, state, device, player = "player_0"):
    #NOTE: Needs to take in a player for the factory stuff?
    obs = obs[player]
    action_mask = torch.ones((48, 48, UNIT_ACTION_IDXS), device=device, dtype=torch.bool)

    #Get unit position
    units = obs["units"][player]    
    valid_actions = 0

    for unit in units.values():
        x, y = unit["pos"]
        action_mask[x, y] = single_unit_action_mask(unit, obs, state, device)
        valid_actions += action_mask[x, y].sum()
    return action_mask


def factory_action_mask(obs, device, player = "player_0"):
    #NOTE: Needs to take in a player for the factory stuff?
    action_mask = torch.ones((48, 48, FACTORY_ACTION_IDXS), device=device, dtype=torch.bool)
    obs = obs[player]
    factories = obs["factories"][player]
    for factory in factories.values():
        x, y = factory["pos"]
        action_mask[x, y] = single_factory_action_mask(factory, obs, device)

    return action_mask
