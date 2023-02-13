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

def create_mask_from_pos(array, x_pos, y_pos, value):
    x = x_pos.flatten()
    y = y_pos.flatten()
    batch_idx = jnp.arange(0, array.shape[0]).repeat(x_pos.shape[-1])

    return array.at[batch_idx, x, y].set(value, mode = "drop")


def _image_features(state, config):
    torch_state = state.to_torch()
    map_size = config["map_size"]
    batch_size = config["parallel_envs"]
    player_0_id = 0
    player_1_id = 1
    
    #NOTE: First channel ALWAYS unit_mask, second channel ALWAYS factory mask
    unit_mask_player_0 = jnp.zeros((batch_size, map_size, map_size))
    unit_mask_player_1 = jnp.zeros((batch_size, map_size, map_size))
    #TODO: We also need a map for where factories are occupying space? Since they are 3x3
    factory_mask_player_0 = jnp.zeros((batch_size, map_size, map_size))
    factory_mask_player_1 = jnp.zeros((batch_size, map_size, map_size))
    #TODO: Add metal/water?
    unit_power = jnp.zeros((batch_size, map_size, map_size))
    unit_ice = jnp.zeros((batch_size, map_size, map_size))
    unit_ore = jnp.zeros((batch_size, map_size, map_size))
    unit_water = jnp.zeros((batch_size, map_size, map_size))
    unit_metal = jnp.zeros((batch_size, map_size, map_size))

    factory_power = jnp.zeros((batch_size, map_size, map_size))
    factory_ice = jnp.zeros((batch_size, map_size, map_size))
    factory_ore = jnp.zeros((batch_size, map_size, map_size))
    factory_water = jnp.zeros((batch_size, map_size, map_size))
    factory_metal = jnp.zeros((batch_size, map_size, map_size))

    factory_cargo = torch.zeros((5, config["map_size"], config["map_size"]))

    board_rubble = state.board.map.rubble
    board_ice = state.board.map.ice 
    board_ore = state.board.map.ore
    board_lichen = state.board.lichen
    
    #TODO: No idea if this works
    player_0_lichen_mask = jnp.isin(
        state.board.lichen_strains, state.factories.team_id[:, player_0_id]
    )
    player_1_lichen_mask = jnp.isin(
        state.board.lichen_strains, state.factories.team_id[:, player_1_id]
    )

    unit_type = torch.zeros((2, config["map_size"], config["map_size"])) #LIGHT, HEAVY

    #UNIT MASKS
    units = state.units
    
    unit_pos_player_0 = units.pos.pos[..., player_0_id, :, :]
    unit_pos_player_1 = units.pos.pos[..., player_1_id, :, :]

    unit_mask_player_0 = create_mask_from_pos(unit_mask_player_0, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], True)
    unit_mask_player_1 = create_mask_from_pos(unit_mask_player_1, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], True)

    #FACTORY MASKS
    factories = state.factories

    factory_pos_player_0 = factories.pos.pos[..., player_0_id, :, :]
    factory_pos_player_1 = factories.pos.pos[..., player_1_id, :, :]

    factory_mask_player_0 = create_mask_from_pos(factory_mask_player_0, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], True)
    factory_mask_player_1 = create_mask_from_pos(factory_mask_player_1, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], True)

    #UNIT CARGO INFO
    #TODO: Test these
    power = units.power
    unit_power = create_mask_from_pos(unit_power, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], power[:, 0].flatten())
    unit_power = create_mask_from_pos(unit_power, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], power[:, 1].flatten()) #Update it

    ice = units.cargo.ice
    unit_ice = create_mask_from_pos(unit_ice, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], ice[:, 0].flatten())
    unit_ice = create_mask_from_pos(unit_ice, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], ice[:, 1].flatten()) #Update it

    ore = units.cargo.ore
    unit_ore = create_mask_from_pos(unit_ore, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], ore[:, 0].flatten())
    unit_ore = create_mask_from_pos(unit_ore, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], ore[:, 1].flatten()) #Update it

    water = units.cargo.water
    unit_water = create_mask_from_pos(unit_water, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], water[:, 0].flatten())
    unit_water = create_mask_from_pos(unit_water, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], water[:, 1].flatten()) #Update it

    metal = units.cargo.metal
    unit_metal = create_mask_from_pos(unit_metal, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], metal[:, 0].flatten())
    unit_metal = create_mask_from_pos(unit_metal, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], metal[:, 1].flatten()) #Update it

    #FACTORY CARGO INFO
    #TODO: Test these
    power = factories.power
    factory_power = create_mask_from_pos(factory_power, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], power[:, 0].flatten())
    factory_power = create_mask_from_pos(factory_power, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], power[:, 1].flatten()) #Update it

    ice = factories.cargo.ice
    factory_ice = create_mask_from_pos(factory_ice, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], ice[:, 0].flatten())
    factory_ice = create_mask_from_pos(factory_ice, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], ice[:, 1].flatten()) #Update it

    ore = factories.cargo.ore
    factory_ore = create_mask_from_pos(factory_ore, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], ore[:, 0].flatten())
    factory_ore = create_mask_from_pos(factory_ore, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], ore[:, 1].flatten()) #Update it

    water = factories.cargo.water
    factory_water = create_mask_from_pos(factory_water, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], water[:, 0].flatten())
    factory_water = create_mask_from_pos(factory_water, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], water[:, 1].flatten()) #Update it

    metal = factories.cargo.metal
    factory_metal = create_mask_from_pos(factory_metal, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], metal[:, 0].flatten())
    factory_metal = create_mask_from_pos(factory_metal, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], metal[:, 1].flatten()) #Update it

    #LIGHT = 0, HEAVY = 1
    unit_type = units.unit_type

    print(unit_type.sum())
    print(unit_type.shape)


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
