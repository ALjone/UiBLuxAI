from math import floor

str_to_dir = {"east": [1, 0], "west": [-1, 0], "north": [0, -1], "south": [0, 1], "center": [0, 0]}

def calculate_move_cost(x, y, base_cost, modifier, rubble, dir, enemy_factory_occupancy_mask):
    if dir.lower() == "center":
        print("????")
    dir = str_to_dir[dir.lower()]
    x, y = x+dir[0], y+dir[1]

    return floor(base_cost + modifier*rubble[x, y]) + enemy_factory_occupancy_mask[x, y]*2000

def can_transfer_to_tile(x, y, factory_occupancy_map, dir):
    dir = str_to_dir[dir.lower()]

    if factory_occupancy_map[x+dir[0], y+dir[1]] == 1:
        return True
    return False