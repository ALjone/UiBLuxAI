import numpy as np
import torch

UNIT_ACTION_IDXS = 7
FACTORY_ACTION_IDXS = 3

def unit_idx_to_action(idx):
    """Translates an index-action (from argmax) into a Lux-valid action for units"""
    assert -1 < idx < 7
    if -1 < idx < 4:
        return move(idx)

    if idx == 5:
        return dig()

    if idx == 6:
        return recharge(20)

def recharge(x, repeat=0, n=1):
    """Gets the action for rechargings"""
    return np.array([5, 0, 0, x, repeat, n])

def dig(repeat=0, n=1):
    """Gets the action for digging"""
    return np.array([3, 0, 0, 0, repeat, n])


def move(dir, repeat = 0, n = 1):
    """Gets the action for moving in a direction"""
    #dir (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    if dir != 0:
        print("Moved a unit!", dir)
    return np.array([0, dir, 0, 0, repeat, n])


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
    unit_output = unit_output.squeeze()
    for unit in units:
            x, y = unit["pos"][0], unit["pos"][1]
            action = torch.argmax(unit_output[x, y]).item()
            actions[unit["unit_id"]] = [unit_idx_to_action(action)]

    return actions

def factory_output_to_actions(factory_output, factories):
    """Turns outputs from the model into action dicts for factories"""
    actions = {}
    factory_output = factory_output.squeeze()
    for factory in factories:
            x, y = factory["pos"][0], factory["pos"][1]
            action = torch.argmax(factory_output[x, y]).item()
            actions[factory["unit_id"]] = 0#factory_idx_to_action(action)

    return actions
