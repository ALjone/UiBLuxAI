import numpy as np
from single_move_actions import move, dig, recharge, self_destruct, transfer

UNIT_ACTION_IDXS = 10
FACTORY_ACTION_IDXS = 4

# a[0] = action type
# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)

# a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)

# a[3] = X, amount of resources transferred or picked up if action is transfer or pickup.
# If action is recharge, it is how much energy to store before executing the next action in queue

# a[4] = repeat. If repeat == 0, then action is not recycled and removed once we have executed it a[5] = n times. 
# Otherwise if repeat > 0 we recycle this action to the back of the action queue and set n = repeat.

# a[5] = n, number of times to execute this action before exhausting it and removing it from the front of the action queue. Minimum is 1.

def unit_idx_to_action(idx, type):
    """Translates an index-action (from argmax) into a Lux-valid action for units"""
    assert -1 < idx < UNIT_ACTION_IDXS
    assert isinstance(idx, int)
    if -1 < idx < 5:
        return move(idx)

    if idx == 5:
        return dig()

    if idx == 6:
        return recharge(100 if type == "LIGHT" else 1000)

    if idx == 7:
         return self_destruct()

    if idx == 8:
        return transfer("ice")
    
    if idx == 9:
        return transfer("ore")


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
            x, y = unit["pos"][0], unit["pos"][1]
            action = unit_output[x, y].item()
            actions[unit["unit_id"]] = [unit_idx_to_action(action, unit["unit_type"])]

    return actions

def factory_output_to_actions(factory_output, factories):
    """Turns outputs from the model into action dicts for factories"""
    actions = {}
    for factory in factories:
            x, y = factory["pos"][0], factory["pos"][1]
            action = factory_output[x, y].item()
            if action == 3:
                 continue
            actions[factory["unit_id"]] = factory_idx_to_action(action)

    return actions
