import numpy as np
from .action_queues import move_north, move_south, move_east, move_west, move_single
from .action_queues import pickup, self_destruct, dig
from .action_queues import move_to_closest_factory_and_transport, move_to_closest_res, res_mining_loop

UNIT_ACTION_IDXS = 13
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

def _unit_idx_to_action(idx, obs, factory_map, unit):
    """Translates an index-action (from argmax) into a Lux-valid action for units"""
    assert 0 < idx < UNIT_ACTION_IDXS
    assert isinstance(idx, int)
    if idx == 0:
        raise ValueError("This shouldn't be possible! Better to just not submit an action")
    if idx == 1: 
        return [move_single(0, 0, 1)] #ABORT QUEUE
    if idx == 2:
        return move_north()
    if idx == 3: 
        return move_south()
    if idx == 4: 
        return move_east()
    if idx == 5:
        return move_west()
    if idx == 6:
        return pickup("power", 1000)
    if idx == 7:
        return move_to_closest_factory_and_transport(factory_map, unit, "ice")
    if idx == 8:
        return move_to_closest_factory_and_transport(factory_map, unit, "ore")
    if idx == 9:
        return move_to_closest_res("ice", unit, obs)
    if idx == 10:
        return move_to_closest_res("ore", unit, obs)
    if idx == 11:
        return res_mining_loop("ice", unit, obs, factory_map)
    if idx == 12:
        return res_mining_loop("ore", unit, obs, factory_map)


def _factory_idx_to_action(idx):
    """Translates an index-action (from argmax) into a Lux-valid action for factories"""
    assert 0 <= idx < (FACTORY_ACTION_IDXS-1) #Minus 1 because action 3 is do nothing
    assert isinstance(idx, int)
    # 0: Build light, 1: Build heavy, 2: Grow lichen
    return idx


def outputs_to_actions(unit_output, factory_output, units, factories, obs, factory_map):
    """Turns outputs from the model into action dicts"""
    unit_actions = _unit_output_to_actions(unit_output, units, obs, factory_map)
    unit_actions.update(_factory_output_to_actions(factory_output, factories))
    return unit_actions

def _unit_output_to_actions(unit_output, units, obs, factory_map):
    """Turns outputs from the model into action dicts for units"""
    actions = {}
    for unit in units:
            x, y = unit["pos"][0], unit["pos"][1]
            action = unit_output[x, y].item()
            if action == 0: continue #The "Do nothing" action
            action = _unit_idx_to_action(action, obs, factory_map, unit)
            if action == []: #If you stand on ice and move to ice...
                continue
            actions[unit["unit_id"]] = action

    return actions

def _factory_output_to_actions(factory_output, factories):
    """Turns outputs from the model into action dicts for factories"""
    actions = {}
    for factory in factories:
            x, y = factory["pos"][0], factory["pos"][1]
            action = factory_output[x, y].item()
            if action == 3: continue
            actions[factory["unit_id"]] = _factory_idx_to_action(action)

    return actions
