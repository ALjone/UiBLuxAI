from .single_move_actions import move as move_single
from .single_move_actions import pickup as pickup_single
from .single_move_actions import dig as dig_single
from .single_move_actions import transfer as transfer_single
from .single_move_actions import self_destruct as self_destruct_single
from utils.utils import find_closest_tile
import numpy as np


# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def move_north():
    return [move_single(1, 1, 1)]

def move_east():
    return [move_single(2, 1, 1)]

def move_south():
    return [move_single(3, 1, 1)]

def move_west():
    return [move_single(4, 1, 1)]

def pickup(res_type, amount):
    return [pickup_single(res_type, amount)]

def self_destruct():
    return [self_destruct_single()]

def dig(unit, part_of_max, repeat):
    """Digs until cargo is filled up to percentage_of_max. Repeat is 0 or 1"""
    cargo = unit["cargo"]
    cargo_space_used = cargo["ice"]+cargo["ore"]+cargo["water"]+cargo["metal"]
    if unit["unit_type"] == "LIGHT":
        cargo_space_left = 100-cargo_space_used #LIGHT has 100 cargo space
        possible_dig_num = cargo_space_left/2   #LIGHT digs 2 per turn

    elif unit["unit_type"] == "HEAVY":
        cargo_space_left = 1000-cargo_space_used #HEAVY has 1000 cargo space
        possible_dig_num = cargo_space_left/20   #HEAVY digs 20 per turn
    
    else:
        raise ValueError("Unit is neither LIGHT nor HEAVY??")
    
    return [dig_single(repeat*int(50*part_of_max), int(possible_dig_num*part_of_max))] #50 for LIGHT and HEAVY

def _move_to_tile(loc_x, loc_y, unit_x, unit_y, repeat):
    """Move to the tile specified. Repeat is 0 or 1"""
    action_queue = []
    x_dir_amount = loc_x-unit_x

    if x_dir_amount > 0:
        action_queue.append(move_single("east", repeat*np.abs(x_dir_amount), np.abs(x_dir_amount)))
    elif x_dir_amount < 0:
        action_queue.append(move_single("west", repeat*np.abs(x_dir_amount), np.abs(x_dir_amount)))

    y_dir_amount = loc_y-unit_y

    if y_dir_amount > 0:
        action_queue.append(move_single("south", repeat*np.abs(y_dir_amount), np.abs(y_dir_amount)))
    elif y_dir_amount < 0:
        action_queue.append(move_single("north", repeat*np.abs(y_dir_amount), np.abs(y_dir_amount)))

    return action_queue

def move_to_closest_res(res_type, unit, obs, repeat = 0)-> list:  
    action_queue = []
    res_map = obs["player_0"]["board"][res_type]

    res_loc_x, res_loc_y = find_closest_tile(res_map, unit["pos"])
    unit_x, unit_y = unit["pos"]

    action_queue += _move_to_tile(res_loc_x, res_loc_y, unit_x, unit_y, repeat)
    return action_queue

def move_to_closest_factory_and_transport(factory_map, unit, res_type, repeat = 0) -> list:
    action_queue = []
    print(factory_map.shape)
    factory_map[factory_map != 1] = 0         #Turn it into actual pos

    factory_loc_x, factory_loc_y = find_closest_tile(factory_map, unit["pos"])
    unit_x, unit_y = unit["pos"]

    action_queue += _move_to_tile(factory_loc_x, factory_loc_y, unit_x, unit_y, repeat)
    if repeat == 1:
        action_queue.append(transfer_single(res_type, 1000, repeat = repeat))
    
    return action_queue

def res_mining_loop(res_type, unit, obs, factory_map):
    action_queue = []

    action_queue += move_to_closest_res(res_type, unit, obs, repeat = 1)
    action_queue += dig(unit, 1, repeat = 1)
    action_queue += move_to_closest_factory_and_transport(factory_map, unit, res_type, repeat = 1)
    
    return action_queue