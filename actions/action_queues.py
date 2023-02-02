from .single_move_actions import move
from .single_move_actions import pickup_single
import numpy as np
from utils.utils import find_closest_ice_tile

x_to_dir = {
            -1 : "west",
             1 : "east"
            }
y_to_dir = {
             -1 : "north",
              1 : "south"
}
# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def move_north():
    return move(1, 1, 1)

def move_east():
    return move(2, 1, 1)

def move_down():
    return move(3, 1, 1)

def move_left():
    return move(4, 1, 1)

def pickup(type, amount):
    pickup_single(type, amount)

def move_to_nearest_ice(unit, obs):
    action_queue = []
    ice_map = obs["player_0"]["board"]["ice"]
    ice_loc_x, ice_loc_y = find_closest_ice_tile(ice_map, unit["pos"])
    unit_x, unit_y = unit["pos"]

    x_dir_amount = ice_loc_x-unit_x
    dir = x_to_dir(np.sign(x_dir_amount))
    x_dir_amount = np.abs(x_dir_amount)

    if x_dir_amount > 0:
        action_queue.append(move(dir, 0, x_dir_amount))

    y_dir_amount = ice_loc_y-unit_y
    dir = y_to_dir(np.sign(y_dir_amount))
    y_dir_amount = np.abs(y_dir_amount)

    if y_dir_amount > 0:
        action_queue.append(move(dir, 0, y_dir_amount))

    