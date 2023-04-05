from actions.single_move_actions import move, move_to_closest_thing, transfer_to_closest_thing, dig, pickup
import numpy as np

MOVES = [   "No move", 
            "Move North",
            "Move East", 
            "Move South", 
            "Move West",
            "Transfer max to closest factory",
            "Dig",
            "Pick up power",
            #"Transport to closest unit",
            ]

UNIT_ACTION_IDXS = len(MOVES)



def _unit_idx_to_action(idx, unit_x, unit_y, state, max_res_idx, rubble):
    assert 0 < idx < UNIT_ACTION_IDXS #Bigger than 0 because 0 should be handled as "Don't submit an action"
    
    #NOTE: Normal MOVE commands
    if 1 <= idx < 5:
        return move(idx, 0, 1)
    
    #NOTE: Transfer
    if idx == 5:
        factory_map = state[2] #NOTE: State[2] is factory occupancy mask for friendly.
        return transfer_to_closest_thing(factory_map, unit_x, unit_y, max_res_idx, rubble)
    
    #NOTE: Dig
    if idx == 6:
        return dig()
    
    #NOTE: Pickup power
    if idx == 7:
        return pickup("power")
    
    raise ValueError("Should've returned by now???")


def unit_output_to_actions(unit_output, units, state, rubble):


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
