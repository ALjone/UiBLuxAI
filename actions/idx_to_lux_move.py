import numpy as np

index_map = np.array(( (0,0,0,0,0,1),
                      
                        (0,1,0,0,0,1),
                        (0,2,0,0,0,1),
                        (0,3,0,0,0,1),
                        (0,4,0,0,0,1),

                        (1,1,0,0,0,1),
                        (1,2,0,0,0,1),
                        (1,3,0,0,0,1),
                        (1,4,0,0,0,1),

                        (1,1,1,1,0,1),
                        (1,2,1,1,0,1),
                        (1,3,1,1,0,1),
                        (1,4,1,1,0,1),

                        (1,1,2,1,0,1),
                        (1,2,2,1,0,1),
                        (1,3,2,1,0,1),
                        (1,4,2,1,0,1),

                        (1,1,4,0,0,1),
                        (1,1,4,1,0,1),
                        (1,1,4,2,0,1),
                        (1,1,4,3,0,1),

                        (1,2,4,0,0,1),
                        (1,2,4,1,0,1),
                        (1,2,4,2,0,1),
                        (1,2,4,3,0,1),

                        (1,3,4,0,0,1),
                        (1,3,4,1,0,1),
                        (1,3,4,2,0,1),
                        (1,3,4,3,0,1),

                        (1,4,4,0,0,1),
                        (1,4,4,1,0,1),
                        (1,4,4,2,0,1),
                        (1,4,4,3,0,1),
                        
                        (2,0,4,1,0,1),
                        (2,0,0,0,0,1),
                        (2,0,1,0,0,1),
                        (2,0,2,0,0,1),
                        (3,0,0,0,0,1),
                        (4,0,0,0,0,1)), dtype = np.int8)


index_map = np.array((  (0,0,0,0,0,1),
                        (0,1,0,0,0,1),
                        (0,2,0,0,0,1),
                        (0,3,0,0,0,1),
                        (0,4,0,0,0,1),
                        
                        (1,1,0,1000,0,1),
                        (1,2,0,1000,0,1),
                        (1,3,0,1000,0,1),
                        (1,4,0,1000,0,1),

                        (2,0,4,1000,0,1),
                        (3,0,0,0,0,1)), dtype = np.int16)


MOVE_NAMES = moves = ["No action", "Move north", "Move east", "Move south", "Move west", "Transport north", "Transport east", "Transport south", "Transport west", "Pickup power", "Dig"]
UNIT_ACTION_IDXS = len(index_map)
FACTORY_ACTION_IDXS = 4 #Light, heavy, water, nothing


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
        action_idx = unit_output[x, y].detach()
        if action_idx == 0: #Do nothing
            continue
        action = index_map[action_idx]
        
        actions[unit["unit_id"]] = [action]

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
