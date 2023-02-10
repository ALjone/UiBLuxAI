import numpy as np

from .single_move_actions import move, dig, recharge, self_destruct, transfer, pickup

UNIT_ACTION_IDXS = 18 # 5 types, 5 directions, 4 values, 4 resources
FACTORY_ACTION_IDXS = 4
MOVE_NAMES = ['Move', 'Transfer', 'Pickup', 'Dig', 'Self Destruct']

# a[0] = action type
# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)

# a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)

# a[3] = X, amount of resources transferred or picked up if action is transfer or pickup.
# If action is recharge, it is how much energy to store before executing the next action in queue

# a[4] = repeat. If repeat == 0, then action is not recycled and removed once we have executed it a[5] = n times.
# Otherwise if repeat > 0 we recycle this action to the back of the action queue and set n = repeat.

# a[5] = n, number of times to execute this action before exhausting it and removing it from the front of the action queue. Minimum is 1.


def _unit_idx_to_action(type_idx, direction_idx, amount_idx, resource_idx, unit):
    """Translates an index-action (from argmax) into a Lux-valid action for units"""

    amount = [0.25, 0.5, 0.75, 1][amount_idx]
    res_name = ["ice", "ore", "water", "metal", "power"][resource_idx]
    if (type_idx == 0):  # Move
        return move(dir = direction_idx)

    elif (type_idx == 1): #Transfer
        cargo = unit["cargo"][res_name]
        return transfer(direction = direction_idx, res_type = resource_idx, amount = int(amount*int(cargo)))

    elif (type_idx == 2): #Pickup
        cargo = unit["cargo"][res_name]
        return pickup(res_type = resource_idx, amount = int(amount*int(cargo)))

    elif (type_idx == 3): #Dig
        return dig()

    elif (type_idx == 4): #Self destruct
        return self_destruct()


def _factory_idx_to_action(idx):
    """Translates an index-action (from argmax) into a Lux-valid action for factories"""
    assert 0 <= idx < (FACTORY_ACTION_IDXS - 1)  # Minus 1 because action 3 is do nothing
    assert isinstance(idx, int)
    # 0: Build light, 1: Build heavy, 2: Grow lichen
    return idx


def outputs_to_actions(unit_output, factory_output, units, factories, obs, factory_map):
    """Turns outputs from the model into action dicts"""
    unit_actions = _unit_output_to_actions(
        unit_output, units, obs, factory_map)
    unit_actions.update(_factory_output_to_actions(factory_output, factories))
    return unit_actions


def _unit_output_to_actions(unit_output, units, obs, factory_map):
    """Turns outputs from the model into action dicts for units"""
    actions = {}
    for unit in units:
        x, y = unit["pos"][0], unit["pos"][1]
        action = unit_output[x, y].detach()#.item()
        action = _unit_idx_to_action(*action, unit)
        actions[unit["unit_id"]] = action

    return actions


def _factory_output_to_actions(factory_output, factories):
    """Turns outputs from the model into action dicts for factories"""
    actions = {}
    for factory in factories:
        x, y = factory["pos"][0], factory["pos"][1]
        action = factory_output[x, y].item()
        if action == 3:
            continue
        actions[factory["unit_id"]] = _factory_idx_to_action(action)

    return actions
