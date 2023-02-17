#from typing import Dicttorcht
import numpy as np
import torch
import gym
import numpy as torch
import numpy.typing as torcht
from gym import spaces
import jax.numpy as jnp
import jux
import jax
from jux.torch import from_torch, to_torch
import jax.scipy.special.entr as jentropy

torch
#Delta change, idx to mapping
#TODO: Triple check this!!!
dirs = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]


def create_mask_from_pos(array, x_pos, y_pos, value):
    x = x_pos.flatten()
    y = y_pos.flatten()
    batch_idx = jnp.arange(0, array.shape[0]).repeat(x_pos.shape[-1])

    return array.at[batch_idx, x, y].set(value, mode = "drop")

jit_create_mask_from_pos = jax.jit(create_mask_from_pos)


def _image_features(state):
    map_size = 48
    batch_size = 50
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

    factory_cargo = jnp.zeros((5, map_size, map_size))

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

    unit_type = jnp.zeros((2, map_size, map_size)) #LIGHT, HEAVY

    #UNIT MASKS
    units = state.units
    
    unit_pos_player_0 = units.pos.pos[..., player_0_id, :, :]
    unit_pos_player_1 = units.pos.pos[..., player_1_id, :, :]

    unit_mask_player_0 = jit_create_mask_from_pos(unit_mask_player_0, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], True)
    unit_mask_player_1 = jit_create_mask_from_pos(unit_mask_player_1, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], True)

    #FACTORY MASKS
    factories = state.factories

    factory_pos_player_0 = factories.pos.pos[..., player_0_id, :, :]
    factory_pos_player_1 = factories.pos.pos[..., player_1_id, :, :]

    factory_mask_player_0 = jit_create_mask_from_pos(factory_mask_player_0, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], True)
    factory_mask_player_1 = jit_create_mask_from_pos(factory_mask_player_1, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], True)

    #UNIT CARGO INFO
    #TODO: Test these
    power = units.power
    unit_power = jit_create_mask_from_pos(unit_power, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], power[:, 0].flatten())
    unit_power = jit_create_mask_from_pos(unit_power, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], power[:, 1].flatten()) #Update it

    ice = units.cargo.ice
    unit_ice = jit_create_mask_from_pos(unit_ice, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], ice[:, 0].flatten())
    unit_ice = jit_create_mask_from_pos(unit_ice, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], ice[:, 1].flatten()) #Update it

    ore = units.cargo.ore
    unit_ore = jit_create_mask_from_pos(unit_ore, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], ore[:, 0].flatten())
    unit_ore = jit_create_mask_from_pos(unit_ore, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], ore[:, 1].flatten()) #Update it

    water = units.cargo.water
    unit_water = jit_create_mask_from_pos(unit_water, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], water[:, 0].flatten())
    unit_water = jit_create_mask_from_pos(unit_water, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], water[:, 1].flatten()) #Update it

    metal = units.cargo.metal
    unit_metal = jit_create_mask_from_pos(unit_metal, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], metal[:, 0].flatten())
    unit_metal = jit_create_mask_from_pos(unit_metal, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], metal[:, 1].flatten()) #Update it

    #FACTORY CARGO INFO
    #TODO: Test these
    power = factories.power
    factory_power = jit_create_mask_from_pos(factory_power, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], power[:, 0].flatten())
    factory_power = jit_create_mask_from_pos(factory_power, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], power[:, 1].flatten()) #Update it

    ice = factories.cargo.ice
    factory_ice = jit_create_mask_from_pos(factory_ice, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], ice[:, 0].flatten())
    factory_ice = jit_create_mask_from_pos(factory_ice, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], ice[:, 1].flatten()) #Update it

    ore = factories.cargo.ore
    factory_ore = jit_create_mask_from_pos(factory_ore, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], ore[:, 0].flatten())
    factory_ore = jit_create_mask_from_pos(factory_ore, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], ore[:, 1].flatten()) #Update it

    water = factories.cargo.water
    factory_water = jit_create_mask_from_pos(factory_water, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], water[:, 0].flatten())
    factory_water = jit_create_mask_from_pos(factory_water, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], water[:, 1].flatten()) #Update it

    metal = factories.cargo.metal
    factory_metal = jit_create_mask_from_pos(factory_metal, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], metal[:, 0].flatten())
    factory_metal = jit_create_mask_from_pos(factory_metal, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], metal[:, 1].flatten()) #Update it

    #LIGHT = 0, HEAVY = 1
    # TODO: Check inplace modification problems with unit_pos_player_X
    friendly_heavy_unit_mask = units.unit_type[:, player_0_id, :] == 1
    enemy_heavy_unit_mask = units.unit_type[:, player_1_id, :] == 1

    heavy_pos_player_0 = jit_create_mask_from_pos(
        jnp.zeros(batch_size, map_size, map_size),
        unit_pos_player_0[..., 0].at[~friendly_heavy_unit_mask].set(127),
        unit_pos_player_0[..., 1].at[~friendly_heavy_unit_mask].set(127)
        )
    light_pos_player_0 = jit_create_mask_from_pos(
        jnp.zeros(batch_size, map_size, map_size),
        unit_pos_player_0[..., 0].at[friendly_heavy_unit_mask].set(127),
        unit_pos_player_0[..., 1].at[friendly_heavy_unit_mask].set(127),
        )

    heavy_pos_player_1 = jit_create_mask_from_pos(
        jnp.zeros(batch_size, map_size, map_size),
        unit_pos_player_1[..., 0].at[~enemy_heavy_unit_mask].set(127),
        unit_pos_player_1[..., 1].at[~enemy_heavy_unit_mask].set(127),
        )
    light_pos_player_1 = jit_create_mask_from_pos(
        jnp.zeros(batch_size, map_size, map_size),
        unit_pos_player_1[..., 0].at[enemy_heavy_unit_mask].set(127),
        unit_pos_player_1[..., 1].at[enemy_heavy_unit_mask].set(127),
        )

    return 1, 2

jit_image_features = jax.jit(_image_features)
print("Jitted")
"""
    print(unit_type.sum())
    print(unit_type.shape)


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
"""
def _global_features(state):
    player_0_id = 0
    player_1_id = 1

    #All these are common
    day = jnp.sin((2*jnp.pi*state.real_env_steps)/1000)*0.3
    night = jnp.cos((2*jnp.pi*state.real_env_steps)/1000)*0.2
    timestep = state.real_env_steps / 1000
    day_night = state.real_env_steps % 50 < 30
    ice_on_map = jnp.sum(state.board.map.ice, (1, 2))
    ore_on_map = jnp.sum(state.board.map.ore, (1, 2))
    ice_entropy = jentropy(state.board.map.ice)
    ore_entropy = jentropy(state.board.map.ore)
    rubble_entropy = jentropy(state.board.map.rubble)
    print(state.n_units)
    print(state.n_units.shape)

    #All these must be flipped
    friendly_factories = state.n_factories[:, player_0_id]
    enemy_factories = state.n_factories[:, player_1_id]
    friendly_light = state.n_units[:, player_0_id, 0]
    friendly_heavy = state.n_units[:, player_0_id, 1]
    enemy_light = state.n_units[:, player_1_id, 0]
    enemy_heavy = state.n_units[:, player_1_id, 1]
    
    #Player 1 lichen
    player_0_lichen_mask = jnp.isin(
        state.board.lichen_strains, state.factories.team_id[:, player_0_id]
    )
    player_1_lichen_mask = jnp.isin(
        state.board.lichen_strains, state.factories.team_id[:, player_0_id]
    )

    friendly_lichen_amount =  jnp.sum(jnp.where(player_0_lichen_mask, state.board.lichen, 0))
    enemy_lichen_amount =  jnp.sum(jnp.where(player_1_lichen_mask, state.board.lichen, 0))
    

    lichen_distribution = (friendly_lichen_amount-enemy_lichen_amount)/max(1, friendly_lichen_amount+enemy_lichen_amount)

    return 1, 2
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

jit_global_features = jax.jit(_global_features)


def observation(state):
    
    

    main_player_image_features, other_player_image_features = jit_image_features(state)
    return
    main_player_global_features, other_player_global_features = jit_global_features(state)
    return 

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
