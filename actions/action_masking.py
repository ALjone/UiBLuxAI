from actions.action_mask_utils import calculate_move_cost, can_transfer_to_tile
from actions.actions import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from actions.single_move_actions import find_dir_to_closest
import numpy as np
from utils.utils import load_config

MASK_EPS = 0
MIN_TRANSFER_AMOUNT = 2
FACTORY_MIN_POWER = 500

config = load_config()
device = config["device"]
map_size = config["map_size"]

MOVES = [   "No move", 
            "Move North",
            "Move East", 
            "Move South", 
            "Move West",
            "Transfer max to closest factory",
            "Dig",
            "Pick up power",
            ]


def single_unit_action_mask(unit, obs, state):
    """Calculates the action mask for one specific unit"""
    action_mask = np.ones(UNIT_ACTION_IDXS, dtype=np.float32)
    unit_mask = state[0]
    factory_occupancy_mask = state[2]
    enemy_factory_occupancy_mask = state[3]
    factory_power = state[7]*500

    #TODO: Look reaaaaaaally close if there are any bugs here

    #(x, y) coordinates of unit
    x, y = unit["pos"]
    #Power remaining for unit, i.e charge
    #TODO: Uncomment to add cost of action queue
    power = unit["power"] #- (1 if unit["unit_type"] == "LIGHT" else 10) #Subtracting the cost of adding to action queue
    #map_sizexmap_size map of how much rubble is in which square
    rubble = obs["board"]["rubble"]
    ice = obs["board"]["ice"]
    ore = obs["board"]["ore"]
    #TODO: !!!!! Should be own lichen eventually, not every lichen
    lichen = obs["board"]["lichen"]
    #How much digging costs for this unit
    dig_cost = 5 if unit["unit_type"] == "LIGHT" else 60 
    #The base cost of moving for this unit
    move_cost_base = 1 if unit["unit_type"] == "LIGHT" else 20
    #The modifier for how much extra it costs to move over each rubble
    rubble_move_modifier = 0.05 if unit["unit_type"] == "LIGHT" else 1
    

    #NOTE: Move
    # (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    # (0, 0) is top left #TODO: Is this true?
    if y == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "north", enemy_factory_occupancy_mask):
        action_mask[1] = MASK_EPS
    if x == map_size-1 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "east", enemy_factory_occupancy_mask):
        action_mask[2] = MASK_EPS
    if y == map_size-1 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "south", enemy_factory_occupancy_mask):
        action_mask[3] = MASK_EPS
    if x == 0 or power < calculate_move_cost(x, y, move_cost_base, rubble_move_modifier, rubble, "west", enemy_factory_occupancy_mask):
        action_mask[4] = MASK_EPS


    #NOTE: Transfer max to closest factory
    factory_dir = find_dir_to_closest(factory_occupancy_mask, x, y, rubble)
    if not can_transfer_to_tile(x, y, factory_occupancy_mask, factory_dir) or \
    (unit["cargo"]["ice"] < MIN_TRANSFER_AMOUNT and unit["cargo"]["ore"] < MIN_TRANSFER_AMOUNT and unit["cargo"]["water"] < MIN_TRANSFER_AMOUNT and unit["cargo"]["metal"] < MIN_TRANSFER_AMOUNT):
        action_mask[5] = MASK_EPS

    #Dig
    if unit["power"] < dig_cost or factory_occupancy_mask[x, y] == 1 or (ice[x, y] == 0 and ore[x, y] == 0 and rubble[x, y] == 0) or lichen[x, y] > 0:
        action_mask[6] = MASK_EPS

    #Power
    #TODO: Check if factory has enough power
    if not can_transfer_to_tile(x, y, factory_occupancy_mask, "center") or factory_power[x, y] < FACTORY_MIN_POWER:
        action_mask[7] = MASK_EPS

    return action_mask


def calculate_water_cost(x, y, obs):
    #TODO Implement this
    #Should be ceil(number of connected and new lichen tiles / 10)
    return 50

    #IDX to action
    #0 -> LIGHT 
    #1 -> HEAVY
    #2 -> LICHEN
    #3 -> NOTHING
def single_factory_action_mask(factory, obs):
    action_mask = np.ones(FACTORY_ACTION_IDXS, dtype=np.float32)
    
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


def unit_action_mask(obs, state, player):
    #NOTE: Needs to take in a player for the factory stuff?
    obs = obs[player]
    action_mask = np.ones((map_size, map_size, UNIT_ACTION_IDXS), dtype=np.float32)

    #Get unit position
    units = obs["units"][player]    

    for unit in units.values():
        x, y = unit["pos"]
        action_mask[x, y] = single_unit_action_mask(unit, obs, state)

        if np.sum(action_mask[x, y]) == 0:
            print("No valid actions for unit:", unit)

    return action_mask

def factory_action_mask(obs, player):
    #NOTE: Needs to take in a player for the factory stuff?
    action_mask = np.ones((map_size, map_size, FACTORY_ACTION_IDXS), dtype=np.float32)
    obs = obs[player]
    factories = obs["factories"][player]
    for id, factory in factories.items():
        x, y = factory["pos"]
        action_mask[x, y] = single_factory_action_mask(factory, obs)


    return action_mask