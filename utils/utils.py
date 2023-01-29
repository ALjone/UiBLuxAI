import numpy as np

UNIT_ACTION_IDXS = 7
FACTORY_ACTION_IDXS = 3

def unit_idx_to_action(idx):
    """Translates an index-action (from argmax) into a Lux-valid action for units"""
    assert -1 < idx < 7
    assert isinstance(idx, int)
    if -1 < idx < 5:
        return move(idx)

    if idx == 5:
        return dig()

    if idx == 6:
         return transfer(100)
    
    #if idx == 7:
    #    return recharge(100)

    
    
def self_destruct():
     return [np.array([4, 0, 0, 0, 0, 1])]

def transfer(x):
     #TODO: This only transfers ice
     return [np.array([1, 0, 0, x, 0, 1])]

def recharge(x, repeat=0, n=1):
    """Gets the action for rechargings"""
    return [np.array([5, 0, 0, x, repeat, n])]

def dig(repeat=0, n=1):
    """Gets the action for digging"""
    return [np.array([3, 0, 0, 0, repeat, n])]


def move(dir, repeat = 0, n = 1):
    """Gets the action for moving in a direction"""
    #dir (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    return [np.array([0, dir, 0, 0, repeat, n])]


def factory_idx_to_action(idx):
    """Translates an index-action (from argmax) into a Lux-valid action for factories"""
    assert -1 < idx < 3
    assert isinstance(idx, int)
    # 0: Build light, 1: Build heavy, 2: Grow lichen
    return idx


def outputs_to_actions(unit_output, factory_output, units, factories):
    """Turns outputs from the model into action dicts"""
    unit_actions = unit_output_to_actions(unit_output, units)
    unit_actions.update(factory_output_to_actions(factory_output, factories))
    return unit_actions

def unit_output_to_actions(unit_output, units):
    """Turns outputs from the model into action dicts for units"""
    actions = {}
    for unit in units:
            print(unit)
            x, y = unit["pos"][0], unit["pos"][1]
            action = unit_output[x, y].item()
            actions[unit["unit_id"]] = unit_idx_to_action(action)

    return actions

def factory_output_to_actions(factory_output, factories):
    """Turns outputs from the model into action dicts for factories"""
    actions = {}
    for factory in factories:
            x, y = factory["pos"][0], factory["pos"][1]
            action = factory_output[x, y].item()
            actions[factory["unit_id"]] = factory_idx_to_action(action)

    return actions
