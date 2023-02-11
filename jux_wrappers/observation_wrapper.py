#from typing import Dicttorcht
import numpy as np
import torch
import gym
import numpy as torch
import numpy.typing as torcht
from gym import spaces
import jax.numpy as jnp
import jux
from jux.torch import from_torch, to_torch
torch
#Delta change, idx to mapping
#TODO: Triple check this!!!
dirs = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]


def _image_features(state, config):
    torch_state = state.to_torch()
    map_size = config["map_size"]
    batch_size = config["parallel_envs"]
    player_0_id = 0
    player_1_id = 1
    
    #NOTE: First channel ALWAYS unit_mask, second channel ALWAYS factory mask

    unit_mask = torch.zeros((1, config["map_size"], config["map_size"]))
    #TODO: We also need a map for where factories are occupying space? Since they are 3x3
    factory_mask = torch.zeros((1, config["map_size"], config["map_size"]))

    unit_cargo = torch.zeros((3, config["map_size"], config["map_size"])) #Power, Ice, Ore
    factory_cargo = torch.zeros((5, config["map_size"], config["map_size"]))

    board = torch.zeros((4, config["map_size"], config["map_size"])) #Rubble, Ice, Ore, Lichen

    lichen_mask = torch.zeros((1, config["map_size"], config["map_size"])) #1 for friendly, 0 for none, -1 for enemy

    unit_type = torch.zeros((2, config["map_size"], config["map_size"])) #LIGHT, HEAVY

    #TODO: Move this to agent
    action_queue_length = torch.zeros((1, config["map_size"], config["map_size"]))

    next_step = torch.zeros((2, config["map_size"], config["map_size"]))
    #units = jux.tree_util.map_to_aval(state.units)
    units = state.units
    
    n_units_player_0 = state.n_units[:, player_0_id]
    n_units_player_1 = state.n_units[:, player_1_id]
    print("N units player 0:", n_units_player_0)

    pos = units.pos.pos[:, player_0_id]
    pos = pos.at[0, 0, 0].set(1)
    pos = pos.at[0, 0, 1].set(1)

    print(pos[..., 0])
    print(pos[..., 0].shape)

    unit_mask_player_0 = np.zeros((batch_size, map_size, map_size))
    unit_mask_player_0 = unit_mask_player_0.at[:, pos[..., 0], pos[..., 1]].set(1, mode = 'drop')
    print(unit_mask_player_0)
    print(unit_mask_player_0.shape)
    #unit_mask_player_0[units.pos.pos[:, player_0_id, :n_units_player_0]] = 1
    #unit_mask_player_1 = units.pos.pos[:, player_1_id, :n_units_player_1]

    '''factories = state["factories"][player]
    units = state["units"][player]

    for _, unit in units.items():
        x, y = unit["pos"]
        unit_mask[0, x, y] = (1 if player == main_player else -1)
        is_light = unit["unit_type"] == "LIGHT"
        if is_light:
            unit_type[0, x, y] = 1
        else:
            unit_type[1, x, y] = 1
        unit_cargo[0, x, y] = unit["power"]/(150 if is_light else 3000)
        unit_cargo[1, x, y] = unit["cargo"]["ice"]/(100 if is_light else 1000)
        unit_cargo[2, x, y] = unit["cargo"]["ore"]/(100 if is_light else 1000)
        
        action_queue_length[0, x, y] = len(unit["action_queue"])/20

        #Predicting next cell for unit
        if len(unit["action_queue"]) > 0:
            act = unit["action_queue"][0]
            if act[0] == 0:
                dir = dirs[act[1]] #Get the direction we're moving
                if x+dir[0] < 0 or x+dir[0] > 47 or y+dir[1] < 0 or y+dir[1] > 47: continue
                next_step[i, x+dir[0], y+dir[1]] = 1
        pass

    for _, factory in factories.items():
        x, y = factory["pos"]
        factory_mask[0, x, y] = (1 if player == main_player else -1)
        #TODO: Look at tanh?
        factory_cargo[0, x, y] = factory["power"]/1000              
        factory_cargo[1, x, y] = factory["cargo"]["ice"]/1000
        factory_cargo[2, x, y] = factory["cargo"]["ore"]/1000
        factory_cargo[3, x, y] = factory["cargo"]["water"]/1000
        factory_cargo[4, x, y] = factory["cargo"]["metal"]/1000

    #Get lichen mask for this player, and add it to the zeros
    if player in state.teams.keys():
        strain_ids = state.teams[player].factory_strains
        agent_lichen_mask = torch.isin(
            state.board.lichen_strains, strain_ids
        )
        lichen_mask += agent_lichen_mask * (1 if player == main_player else -1)'''

    board[0] = state["board"]["rubble"]/env.state.env_cfg.MAX_RUBBLE
    board[1] = state["board"]["ice"]
    board[2] = state["board"]["ore"]
    board[3] = state["board"]["lichen"]/env.state.env_cfg.MAX_LICHEN_PER_TILE
    
    #TODO: Add action queue type in RL agent
    #Don't ask why this is torch to torch...
    image_features = torch.tensor(torch.concatenate([
        unit_mask,
        factory_mask,
        unit_cargo,
        factory_cargo,
        board,
        lichen_mask,
        unit_type,
        action_queue_length,
        next_step,
    ]))

    image_features_flipped = image_features.clone()
    image_features_flipped[(0, 1, 14)] *= -1

    return image_features.to(torch.float32), image_features_flipped.to(torch.float32)

def _global_features(self, obs):
    main_player = agents[0]
    other_player = agents[1]
    shared_obs = obs[main_player]


    #All these are common
    day = torch.sin((2*torch.pi*env.state.real_env_steps)/1000)*0.3
    night = torch.cos((2*torch.pi*env.state.real_env_steps)/1000)*0.2
    timestep = env.state.real_env_steps / 1000
    day_night = (1 if env.state.real_env_steps % 50 < 30 else 0)
    ice_on_map = torch.sum(shared_obs["board"]["ice"])
    ore_on_map = torch.sum(shared_obs["board"]["ore"])
    ice_entropy = 0
    ore_entropy = 0
    rubble_entropy = 0

    #All these must be flipped
    friendly_factories = len(shared_obs["factories"][main_player].keys())
    enemy_factories = len(shared_obs["factories"][other_player].keys())
    friendly_light = len([unit for unit in shared_obs["units"][main_player].values() if unit["unit_type"] == "LIGHT"])
    friendly_heavy = len([unit for unit in shared_obs["units"][main_player].values() if unit["unit_type"] == "HEAVY"])
    enemy_light = len([unit for unit in shared_obs["units"][other_player].values() if unit["unit_type"] == "LIGHT"])
    enemy_heavy = len([unit for unit in shared_obs["units"][other_player].values() if unit["unit_type"] == "HEAVY"])
    
    #Player 1 lichen
    if main_player in state.teams.keys():
        strain_ids = state.teams[main_player].factory_strains
        agent_lichen_mask = torch.isin(
            state.board.lichen_strains, strain_ids
        )
        friendly_lichen_amount =  torch.sum(shared_obs["board"]["lichen"]*agent_lichen_mask)

        strain_ids = state.teams[other_player].factory_strains
        agent_lichen_mask = torch.isin(
            state.board.lichen_strains, strain_ids
        )
        enemy_lichen_amount =  torch.sum(shared_obs["board"]["lichen"]*agent_lichen_mask)
        if friendly_lichen_amount + enemy_lichen_amount == 0:
            lichen_distribution = 0
        else:
            lichen_distribution = 2*(friendly_lichen_amount/(friendly_lichen_amount+enemy_lichen_amount))-1
    else:
        lichen_distribution = 0

    #TODO: Double check
    main_player_vars = torch.tensor((day, 
                        night, 
                        timestep, 
                        day_night, 
                        ice_on_map, 
                        ore_on_map, 
                        ice_entropy,
                        ore_entropy, 
                        rubble_entropy, 
                        friendly_factories, 
                        friendly_light, 
                        friendly_heavy, 
                        enemy_factories, 
                        enemy_light, 
                        enemy_heavy, 
                        lichen_distribution)
                        )
    
    other_player_vars = torch.tensor((day, 
                        night, 
                        timestep, 
                        day_night, 
                        ice_on_map, 
                        ore_on_map, 
                        ice_entropy,
                        ore_entropy, 
                        rubble_entropy, 
                        enemy_factories, 
                        enemy_light, 
                        enemy_heavy, 
                        friendly_factories, 
                        friendly_light, 
                        friendly_heavy, 
                        lichen_distribution * -1)
                        )


    return main_player_vars.to(torch.float32), other_player_vars.to(torch.float32)


def observation(state, config):
    
    

    
    main_player_image_features, other_player_image_features = _image_features(state, config)

    main_player_global_features, other_player_global_features = global_features(obs)

    #TODO: Add global features
    image_state = {
                    main_player: 
                        {
                            "image_features": main_player_image_features,
                            "global_features": main_player_global_features
                        },
                    other_player: 
                        {
                            "image_features": other_player_image_features,
                            "global_features": other_player_global_features
                        }
                }

    return image_state, obs
