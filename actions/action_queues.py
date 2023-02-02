from .single_move_actions import move
from .single_move_actions import pickup_single
import numpy as np

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