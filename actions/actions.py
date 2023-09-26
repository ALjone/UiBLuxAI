from actions.single_move_actions import move, move_to_closest_thing, transfer_to_closest_thing, dig, pickup
from actions.action_utils import where_will_unit_end_up
import numpy as np

MOVES = [   "No move", 
            "Move North",
            "Move East", 
            "Move South", 
            "Move West",
            "Transfer max to closest factory",
            "Dig",
            "Pick up power",
            #"ore",
            #"ice",
            #"factory"
            #"Transfer to closest unit",
            ]

UNIT_ACTION_IDXS = len(MOVES)
FACTORY_ACTION_IDXS = 4



def _unit_idx_to_action(idx, unit_x, unit_y, state, max_res_idx, rubble, ice, ore):
    assert 0 < idx < UNIT_ACTION_IDXS #Bigger than 0 because 0 should be handled as "Don't submit an action"
    factory_map = state[2] #NOTE: State[2] is factory occupancy mask for friendly.
    
    #NOTE: Normal MOVE commands
    if 1 <= idx < 5:
        return move(idx, 0, 1)
    
    #NOTE: Transfer
    if idx == 5:
        return transfer_to_closest_thing(factory_map, unit_x, unit_y, max_res_idx, rubble)
    
    #NOTE: Dig
    if idx == 6:
        return dig()
    
    #NOTE: Pickup power
    if idx == 7:
        return pickup("power")
    
    if idx == 8:
        return move_to_closest_thing(ice, unit_x, unit_y, rubble)
    if idx == 9:
        return move_to_closest_thing(ore, unit_x, unit_y, rubble)
    if idx == 10:
        return move_to_closest_thing(factory_map, unit_x, unit_y, rubble)
    
    raise ValueError("Should've returned by now???")


def do_stuff_depending_on_action_idx(unit, unit_output, state, rubble, ice, ore, pos_to_units_there, actions):
    x, y = unit["pos"][0], unit["pos"][1]
    cargo = [unit["cargo"]["ice"], unit["cargo"]["ore"], unit["cargo"]["water"], unit["cargo"]["metal"]]
    max_res_idx = np.argmax(cargo).item()
    action_idx = unit_output[x, y].item()
    if action_idx != 0: #Do nothing if == 0
        action = _unit_idx_to_action(action_idx, x, y, state, max_res_idx, rubble, ice, ore)
        new_pos = where_will_unit_end_up(action, x, y)
    else:
        new_pos = (x, y)

    if new_pos in pos_to_units_there.keys():
        pos_to_units_there[(x, y)] = [unit["unit_id"]]#.append(unit["unit_id"])
    else:
        pos_to_units_there[new_pos] = [unit["unit_id"]]
        #TODO: Can be solved better, we shouldn't have to check this idx twice
        if action_idx != 0:
            actions[unit["unit_id"]] = [action]

def unit_output_to_actions(unit_output, units, state, rubble, ice, ore):
    """Turns outputs from the model into action dicts for units"""
    pos_to_units_there = {}
    actions = {}
    units_processed = set()
    for unit in units:
        x, y = unit["pos"][0], unit["pos"][1]
        action_idx = unit_output[x, y].item()
        cargo = [unit["cargo"]["ice"], unit["cargo"]["ore"], unit["cargo"]["water"], unit["cargo"]["metal"]]
        max_res_idx = np.argmax(cargo).item()
        if action_idx == 0 or where_will_unit_end_up(_unit_idx_to_action(action_idx, x, y, state, max_res_idx, rubble, ice, ore), x, y) == (x, y):
            do_stuff_depending_on_action_idx(unit, unit_output, state, rubble, ice, ore, pos_to_units_there, actions)
            units_processed.add(unit["unit_id"])

    for unit in units:
        if unit["unit_id"] in units_processed: continue

        do_stuff_depending_on_action_idx(unit, unit_output, state, rubble, ice, ore, pos_to_units_there, actions)

    #for new_pos, units in pos_to_units_there.items():

        #print(f"Pos: {new_pos} Units: {units}")

    return actions

def factory_output_to_actions(factory_output, factories):
    """Turns outputs from the model into action dicts for factories"""
    actions = {}
    for factory in factories:
        x, y = factory["pos"][0], factory["pos"][1]
        #print("Factory x, y", x, y)
        action = factory_output[x, y].item()
        if action == 3:
            continue
        actions[factory["unit_id"]] = action

    return actions