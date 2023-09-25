from math import floor

str_to_dir = {"east": [1, 0], "west": [-1, 0], "north": [0, -1], "south": [0, 1], "center": [0, 0]}

int_to_dir = {2: [1, 0], 4: [-1, 0], 1: [0, -1], 3: [0, 1], 0: [0, 0]}

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



def where_will_unit_end_up(move, unit_x, unit_y):
    if isinstance(move, list):
        assert len(move) == 1, f"Move is a list, expected it to have length 1, found lenght {len(move)}"
        move = move[0]
    if move[0] != 0:
        return (unit_x, unit_y)
    dir_x, dir_y = int_to_dir[move[1]] #Move[1] is the direction as an int
    return (unit_x+dir_x, unit_y+dir_y)