import numpy as np


def pickup(res_type, amount, repeat = 0, n = 1):
    return [np.array([2, 0, res_type, amount, repeat, n])]


def transfer(direction, res_type, amount, repeat=0, n = 1):
    return [np.array([1, direction, res_type, amount, repeat, n])]


def self_destruct():
    """Gets the action for self-destruction :((("""
    return [np.array([4, 0, 0, 0, 0, 1]).astype(int)]


def recharge(amount, repeat=0, n=1):
    """Gets the action for rechargings"""
    return [np.array([5, 0, 0, amount, repeat, n]).astype(int)]


def dig(repeat=0, n=1):
    """Gets the action for digging"""
    return [np.array([3, 0, 0, 0, repeat, n]).astype(int)]


def move(dir, repeat=0, n=1):
    """Gets the action for moving in a direction"""
    # dir (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    return [np.array([0, dir, 0, 0, repeat, n]).astype(int)]
