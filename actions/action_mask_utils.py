from math import floor

def calculate_move_cost(x, y, base_cost, modifier, rubble, dir, enemy_factory_occupancy_mask):
    dir = dir.lower()
    if dir == "center":
        return 1000 #We don't want to allow move center, because that's the same as do nothing
    if dir == "east":
        dir = [1, 0]
    elif dir == "west":
        dir = [-1, 0]
    elif dir == "north":
        dir = [0, -1]
    elif dir == "south":
        dir = [0, 1]
    else:
        raise ValueError("Expected dir to be a direction, found:", dir)
    x, y = x+dir[0], y+dir[1]

    return floor(base_cost + modifier*rubble[x, y]) + enemy_factory_occupancy_mask[x, y]*1000

def can_transfer_to_tile(x, y, factory_occupancy_map, dir):
    dir = dir.lower()
    if dir == "east":
        dir = [1, 0]
    elif dir == "west":
        dir = [-1, 0]
    elif dir == "north":
        dir = [0, -1]
    elif dir == "south":
        dir = [0, 1]
    elif dir == "center":
        dir = [0, 0]
    else:
        raise ValueError("Expected dir to be a direction, found:", dir)
    #TODO: There are ways to predict where a unit will be next time step, so implement that in the very distant future
    #unit_pos = [list(elem) for elem in unit_pos] #Sometimes this is a numpy array...

    if factory_occupancy_map[x+dir[0], y+dir[1]] == 1:
        return True
    return False