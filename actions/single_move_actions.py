import numpy as np
from utils.utils import find_closest_tile

def pickup(res_type, x=1000):
    if res_type == "ice":
        return np.array([2, 0, 0, x, 0, 1])
    elif res_type == "ore":
        return np.array([2, 0, 1, x, 0, 1])
    elif res_type == "water":
        return np.array([2, 0, 2, x, 0, 1])
    elif res_type == "metal":
        return np.array([2, 0, 3, x, 0, 1])
    elif res_type == "power":
        return np.array([2, 0, 4, x, 0, 1])
    else:
        raise ValueError("Unknown type found:", res_type)


def transfer(res_type, x=1000, direction=None, repeat=0):
    """Transfers all the res to center"""
    assert isinstance(direction, int), f"Expected direction to be int, found {direction}"
    if isinstance(res_type, int):
        return np.array([1, direction, res_type, x, 0, 1])
    if res_type == "ice":
        return np.array([1, direction, 0, x, 0, 1])
    elif res_type == "ore":
        return np.array([1, direction, 1, x, 0, 1])
    elif res_type == "water":
        return np.array([1, direction, 2, x, 0, 1])
    elif res_type == "metal":
        return np.array([1, direction, 3, x, 0, 1])
    elif res_type == "power":
        return np.array([1, direction, 4, x, 0, 1])
    else:
        raise ValueError("Unknown res type found:", res_type)


def self_destruct():
    """Gets the action for self-destruction :((("""
    return np.array([4, 0, 0, 0, 0, 1])


def recharge(x, repeat=0, n=1):
    """Gets the action for rechargings"""
    return np.array([5, 0, 0, x, repeat, 1])


def dig(repeat=0, n=1):
    """Gets the action for digging"""
    return np.array([3, 0, 0, 0, 0, 1]).astype(int)


def move(dir, repeat=0, n=1):
    """Gets the action for moving in a direction"""
    # dir (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    if isinstance(dir, int):
        return np.array([0, dir, 0, 0, 0, 1])
    dir = dir.lower()
    if dir == "center":
        return np.array([0, 0, 0, 0, 0, 1])
    elif dir == "north":
        return np.array([0, 1, 0, 0, 0, 1])
    elif dir == "east":
        return np.array([0, 2, 0, 0, 0, 1])
    elif dir == "south":
        return np.array([0, 3, 0, 0, 0, 1])
    elif dir == "west":
        return np.array([0, 4, 0, 0, 0, 1])
    else:
        raise ValueError("Unknown direction found:", dir)

def _move_to_tile(loc_x, loc_y, unit_x, unit_y, rubble):
    """Move to the tile specified. Repeat is 0 or 1"""
    #TODO: Have a random argmax. Now it's biased in the North/South direction
    best_move = ["Center", 0] #Dir, cost
    x_dir_amount = loc_x-unit_x

    if x_dir_amount > 0:
        best_move = ["East", rubble[unit_x+1, unit_y]]
    elif x_dir_amount < 0:
        best_move = ["West", rubble[unit_x-1, unit_y]]

    y_dir_amount = loc_y-unit_y

    #TODO: Double check if this should be + or - 1
    if y_dir_amount > 0 and (best_move[0] != "Center" or rubble[unit_x, unit_y+1] < best_move[1]):
        best_move = ["South", rubble[unit_x, unit_y+1]]
    elif y_dir_amount < 0 and (best_move[0] != "Center" or rubble[unit_x, unit_y-1] < best_move[1]):
        best_move = ["North", rubble[unit_x, unit_y-1]]

    return best_move[0]


def find_dir_to_closest(res_map, unit_x, unit_y, rubble):
    res_map = np.array(res_map)

    res_loc_x, res_loc_y = find_closest_tile(res_map, (unit_x, unit_y))
    dir = _move_to_tile(res_loc_x, res_loc_y, unit_x, unit_y, rubble)

    return dir

def move_to_closest_thing(res_map, unit_x, unit_y, rubble):  

    dir = find_dir_to_closest(res_map, unit_x, unit_y, rubble)

    if dir.lower() == "center":
        return dig()
    
    return move(dir, 0, 1)
   

def transfer_to_closest_thing(res_map, unit_x, unit_y, res_idx, rubble):
    dir = find_dir_to_closest(res_map, unit_x, unit_y, rubble)
    dir = ["Center", "North", "East", "South", "West"].index(dir)
    
    return transfer(res_idx, 2000, dir, 0)