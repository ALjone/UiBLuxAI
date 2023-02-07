import numpy as np


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


def transfer(res_type, x=1000, direction=1, repeat=0):
    """Transfers all the res to center"""
    if res_type == "ice":
        return np.array([direction, 0, 0, x, repeat, 1])
    elif res_type == "ore":
        return np.array([direction, 0, 1, x, repeat, 1])
    elif res_type == " water":
        return np.array([direction, 0, 2, x, repeat, 1])
    elif res_type == "metal":
        return np.array([direction, 0, 3, x, repeat, 1])
    elif res_type == "power":
        return np.array([direction, 0, 4, x, repeat, 1])
    else:
        raise ValueError("Unknown res type found:", res_type)


def self_destruct():
    """Gets the action for self-destruction :((("""
    return np.array([4, 0, 0, 0, 0, 1])


def recharge(x, repeat=0, n=1):
    """Gets the action for rechargings"""
    return np.array([5, 0, 0, x, repeat, n])


def dig(repeat=0, n=1):
    """Gets the action for digging"""
    return np.array([3, 0, 0, 0, repeat, n])


def move(dir, repeat=0, n=1):
    """Gets the action for moving in a direction"""
    # dir (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    if isinstance(dir, int):
        return np.array([0, dir, 0, 0, repeat, n])
    if dir == "center":
        return np.array([0, 0, 0, 0, repeat, n])
    elif dir == "north":
        return np.array([0, 1, 0, 0, repeat, n])
    elif dir == "east":
        return np.array([0, 2, 0, 0, repeat, n])
    elif dir == "south":
        return np.array([0, 3, 0, 0, repeat, n])
    elif dir == "west":
        return np.array([0, 4, 0, 0, repeat, n])
    else:
        raise ValueError("Unknown direction found:", dir)
