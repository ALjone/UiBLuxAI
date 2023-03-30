from actions.single_move_actions import move, move_to_closest_thing, transfer_to_closest_thing, dig, pickup
import numpy as np
import torch
MOVES = [   "No move", 
            "Move North",
            "Move East", 
            "Move South", 
            "Move West",
            "Move closest ice",
            "Move closest ore",
            "Move closest rubble",
            #Move to closest enemy lichen
            "Move closest factory",
            "Move closest enemy unit",
            "Transfer max to closest factory",
            "Dig",
            "Pick up power",
            #"Transport to closest unit",
            ]
UNIT_ACTION_IDXS = len(MOVES)
FACTORY_ACTION_IDXS = 4 #Light, heavy, water, nothing



def _unit_idx_to_action(idx, unit_x, unit_y, state, max_res_idx, rubble):
    assert 0 < idx < UNIT_ACTION_IDXS #Bigger than 0 because 0 should be handled as "Don't submit an action"
    
    #NOTE: Normal MOVE commands
    if 1 <= idx < 5:
        return move(idx, 0, 1)
    
    #NOTE: Move to
    #Res map goes: Rubble, ice, ore, lichen, i.e 16, 17, 18, 19
    if idx == 7:
        res_map = state[16]
        return move_to_closest_thing(res_map, unit_x, unit_y, rubble)
    if idx == 5:
        res_map = state[17]
        return move_to_closest_thing(res_map, unit_x, unit_y, rubble)
    if idx == 6:
        res_map = state[18]
        return move_to_closest_thing(res_map, unit_x, unit_y, rubble)
    if idx == 8:
        factory_map = state[4] #NOTE: State[4] is factory occupancy mask for friendly.
        return move_to_closest_thing(factory_map, unit_x, unit_y, rubble)
    
    #NOTE: Move to enemy
    if idx == 9:
        if state[2].sum() >= 1: #Must exist at least one res for this
            res_map = state[2] #State[2] is always enemy unit map
            return move_to_closest_thing(res_map, unit_x, unit_y, rubble)
        return move(0, 0, 1) #Default to this 

    #NOTE: Transfer
    if idx == 10:
        factory_map = state[4] #NOTE: State[4] is factory occupancy mask for friendly.
        return transfer_to_closest_thing(factory_map, unit_x, unit_y, max_res_idx, rubble)
    
    #NOTE: Dig
    if idx == 11:
        return dig()
    
    #NOTE: Pickup power
    if idx == 12:
        return pickup("power")
    
    raise ValueError("Should've returned by now???")

def _factory_idx_to_action(idx):
    """Translates an index-action (from argmax) into a Lux-valid action for factories"""
    assert 0 <= idx < (FACTORY_ACTION_IDXS - 1), f"Factory idx makes no sense, {idx}"# Minus 1 because action 3 is do nothing
    assert isinstance(idx, int), "No int???"
    # 0: Build light, 1: Build heavy, 2: Grow lichen
    return idx


def _unit_output_to_actions(unit_output, units, state, rubble):


    """Turns outputs from the model into action dicts for units"""
    actions = {}
    for unit in units:
        x, y = unit["pos"][0], unit["pos"][1]
        cargo = [unit["cargo"]["ice"], unit["cargo"]["ore"], unit["cargo"]["water"], unit["cargo"]["metal"]]
        max_res_idx = np.argmax(cargo).item()
        action_idx = unit_output[x, y].item()
        if action_idx == 0: #Do nothing
            continue
        action = _unit_idx_to_action(action_idx, x, y, state, max_res_idx, rubble)
        
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



def outputs_to_actions(unit_output, factory_output, units, factories, state, obs):
    """Turns outputs from the model into action dicts"""
    rubble = obs["board"]["rubble"]
    unit_actions = _unit_output_to_actions(
        unit_output, units, state, rubble)
    unit_actions.update(_factory_output_to_actions(factory_output, factories))
    return unit_actions
