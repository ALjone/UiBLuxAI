from actions.single_move_actions import move, transfer_all_res_to_dir, dig, pickup
import numpy as np

MOVES = [   "No move", 
            "Move North",
            "Move East", 
            "Move South", 
            "Move West",
            #There's never a use for transfer max center. Almost(?)
            "Transfer max North",
            "Transfer max East",
            "Transfer max South",
            "Transfer max West",
            #"Transfer max to closest factory",
            "Dig",
            "Pick up power",
            #"ore",
            #"ice",
            #"factory"
            #"Transfer to closest unit",
            ]

UNIT_ACTION_IDXS = len(MOVES)
FACTORY_ACTION_IDXS = 4



def _unit_idx_to_action(idx, max_res_idx):
    assert 0 < idx < UNIT_ACTION_IDXS #Bigger than 0 because 0 should be handled as "Don't submit an action"
    
    #NOTE: Normal MOVE commands
    if 1 <= idx < 5:
        return move(idx, 0, 1)
    
    #NOTE: Transfer
    if idx == 5:
        return transfer_all_res_to_dir("north", max_res_idx)
    if idx == 6:
        return transfer_all_res_to_dir("east", max_res_idx)
    if idx == 7:
        return transfer_all_res_to_dir("south", max_res_idx)
    if idx == 8:
        return transfer_all_res_to_dir("west", max_res_idx)

    #NOTE: Dig
    if idx == 9:
        return dig()
    
    #NOTE: Pickup power
    if idx == 10:
        return pickup("power")

    raise ValueError("Should've returned by now???")




def unit_output_to_actions(light_unit_output, heavy_unit_output, units):
    actions = {}
    
    for unit in units:
        unit_output = light_unit_output if unit["unit_type"] == "LIGHT" else heavy_unit_output
        x, y = unit["pos"][0], unit["pos"][1]
        cargo = [unit["cargo"]["ice"], unit["cargo"]["ore"], unit["cargo"]["water"], unit["cargo"]["metal"]]
        max_res_idx = np.argmax(cargo).item()
        action_idx = unit_output[x, y].item()

        if action_idx != 0:  # Do nothing if == 0
            action = _unit_idx_to_action(action_idx, max_res_idx)
            actions[unit["unit_id"]] = [action]
    
    return actions


def factory_output_to_actions(factory_output, factories):
    """Turns outputs from the model into action dicts for factories"""
    actions = {}
    for factory in factories:
        x, y = factory["pos"][0], factory["pos"][1]
        action = factory_output[x, y].item()
        if action == 3:
            continue
        actions[factory["unit_id"]] = action

    return actions
