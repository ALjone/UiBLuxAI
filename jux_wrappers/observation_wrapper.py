import numpy as np
import jax.numpy as jnp
import jux
import jax
from jux.torch import from_torch, to_torch
import jax.scipy.special as js
from utils.utils import load_config


def create_mask_from_pos(array, x_pos, y_pos, value):
    x = x_pos.flatten()
    y = y_pos.flatten()
    batch_idx = jnp.arange(0, array.shape[0]).repeat(x_pos.shape[-1])

    return array.at[batch_idx, x, y].set(value, mode = "drop")

#TODO: Move to utils
jit_create_mask_from_pos = jax.jit(create_mask_from_pos)

class StateProcessor:

    def __init__(self, config) -> None:
        
        self.num_envs = config["parallel_envs"]
        self.map_size = config["map_size"]

        self.create_mask_from_pos = jax.jit(create_mask_from_pos)
        self._global_features = jax.jit(self._global_features)
        self._image_features = jax.jit(self._image_features)
        #self.reward = jax.jit(self.reward)

        self.reset()

    def reset(self):
        self.previous_unit_ore = [jnp.ones((self.num_envs, self.map_size, self.map_size)), jnp.ones((self.num_envs, self.map_size, self.map_size))]
        self.previous_factory_ore = [jnp.ones((self.num_envs, self.map_size, self.map_size)), jnp.ones((self.num_envs, self.map_size, self.map_size))]
        self.previous_unit_ice = [jnp.ones((self.num_envs, self.map_size, self.map_size)), jnp.ones((self.num_envs, self.map_size, self.map_size))]
        self.previous_factory_ice = [jnp.ones((self.num_envs, self.map_size, self.map_size)), jnp.ones((self.num_envs, self.map_size, self.map_size))]

        self.previous_light_units = [0, 0]
        self.previous_heavy_units = [0, 0]


    def _image_features(self, state):
        player_0_id = 0
        player_1_id = 1
        

        #NOTE: First channel ALWAYS unit_mask, second channel ALWAYS factory mask
        unit_mask_player_0 = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        unit_mask_player_1 = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        #TODO: We also need a map for where factories are occupying space? Since they are 3x3
        factory_mask_player_0 = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        factory_mask_player_1 = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        #TODO: Add metal/water?
        unit_power = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        unit_ice = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        unit_ore = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        unit_water = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        unit_metal = jnp.zeros((self.num_envs, self.map_size, self.map_size))

        factory_power = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        factory_ice = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        factory_ore = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        factory_water = jnp.zeros((self.num_envs, self.map_size, self.map_size))
        factory_metal = jnp.zeros((self.num_envs, self.map_size, self.map_size))

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

        #UNIT MASKS
        units = state.units
        
        unit_pos_player_0 = units.pos.pos[..., player_0_id, :, :]
        unit_pos_player_1 = units.pos.pos[..., player_1_id, :, :]


        unit_mask_player_0 = self.create_mask_from_pos(unit_mask_player_0, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], True)
        unit_mask_player_1 = self.create_mask_from_pos(unit_mask_player_1, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], True)

        #FACTORY MASKS
        factories = state.factories

        factory_pos_player_0 = factories.pos.pos[..., player_0_id, :, :]
        factory_pos_player_1 = factories.pos.pos[..., player_1_id, :, :]

        factory_mask_player_0 = self.create_mask_from_pos(factory_mask_player_0, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], True)
        factory_mask_player_1 = self.create_mask_from_pos(factory_mask_player_1, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], True)

        #UNIT CARGO INFO
        #TODO: Test these
        power = units.power
        unit_power = self.create_mask_from_pos(unit_power, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], power[:, 0].flatten())
        unit_power = self.create_mask_from_pos(unit_power, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], power[:, 1].flatten()) #Update it

        ice = units.cargo.ice
        unit_ice = self.create_mask_from_pos(unit_ice, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], ice[:, 0].flatten())
        unit_ice = self.create_mask_from_pos(unit_ice, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], ice[:, 1].flatten()) #Update it

        ore = units.cargo.ore
        unit_ore = self.create_mask_from_pos(unit_ore, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], ore[:, 0].flatten())
        unit_ore = self.create_mask_from_pos(unit_ore, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], ore[:, 1].flatten()) #Update it

        water = units.cargo.water
        unit_water = self.create_mask_from_pos(unit_water, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], water[:, 0].flatten())
        unit_water = self.create_mask_from_pos(unit_water, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], water[:, 1].flatten()) #Update it

        metal = units.cargo.metal
        unit_metal = self.create_mask_from_pos(unit_metal, unit_pos_player_0[..., 0], unit_pos_player_0[..., 1], metal[:, 0].flatten())
        unit_metal = self.create_mask_from_pos(unit_metal, unit_pos_player_1[..., 0], unit_pos_player_1[..., 1], metal[:, 1].flatten()) #Update it

        #FACTORY CARGO INFO
        #TODO: Test these
        power = factories.power
        factory_power = self.create_mask_from_pos(factory_power, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], power[:, 0].flatten())
        factory_power = self.create_mask_from_pos(factory_power, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], power[:, 1].flatten()) #Update it

        ice = factories.cargo.ice
        factory_ice = self.create_mask_from_pos(factory_ice, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], ice[:, 0].flatten())
        factory_ice = self.create_mask_from_pos(factory_ice, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], ice[:, 1].flatten()) #Update it

        ore = factories.cargo.ore
        factory_ore = self.create_mask_from_pos(factory_ore, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], ore[:, 0].flatten())
        factory_ore = self.create_mask_from_pos(factory_ore, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], ore[:, 1].flatten()) #Update it

        water = factories.cargo.water
        factory_water = self.create_mask_from_pos(factory_water, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], water[:, 0].flatten())
        factory_water = self.create_mask_from_pos(factory_water, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], water[:, 1].flatten()) #Update it

        metal = factories.cargo.metal
        factory_metal = self.create_mask_from_pos(factory_metal, factory_pos_player_0[..., 0], factory_pos_player_0[..., 1], metal[:, 0].flatten())
        factory_metal = self.create_mask_from_pos(factory_metal, factory_pos_player_1[..., 0], factory_pos_player_1[..., 1], metal[:, 1].flatten()) #Update it

        #LIGHT = 0, HEAVY = 1
        # TODO: Check inplace modification problems with unit_pos_player_X
        friendly_heavy_unit_mask = units.unit_type[:, player_0_id, :]
        enemy_heavy_unit_mask = units.unit_type[:, player_1_id, :]

        heavy_pos_player_0 = self.create_mask_from_pos(
            jnp.zeros(shape = (self.num_envs, self.map_size, self.map_size)),
            unit_pos_player_0[..., 0].at[~friendly_heavy_unit_mask].set(127),
            unit_pos_player_0[..., 1].at[~friendly_heavy_unit_mask].set(127),
            value = 1
            )
        p0_num_heavy = heavy_pos_player_0.sum((1, 2))
        light_pos_player_0 = self.create_mask_from_pos(
            jnp.zeros(shape = (self.num_envs, self.map_size, self.map_size)),
            unit_pos_player_0[..., 0].at[friendly_heavy_unit_mask].set(127),
            unit_pos_player_0[..., 1].at[friendly_heavy_unit_mask].set(127),
            value = 1
            )
        p0_num_light = light_pos_player_0.sum((1, 2))
        heavy_pos_player_1 = self.create_mask_from_pos(
            jnp.zeros(shape = (self.num_envs, self.map_size, self.map_size)),
            unit_pos_player_1[..., 0].at[~enemy_heavy_unit_mask].set(127),
            unit_pos_player_1[..., 1].at[~enemy_heavy_unit_mask].set(127),
            value = 1
            )
        p1_num_heavy = heavy_pos_player_1.sum((1, 2))
        light_pos_player_1 = self.create_mask_from_pos(
            jnp.zeros(shape = (self.num_envs, self.map_size, self.map_size)),
            unit_pos_player_1[..., 0].at[enemy_heavy_unit_mask].set(127),
            unit_pos_player_1[..., 1].at[enemy_heavy_unit_mask].set(127),
            value = 1
            )
        p1_num_light = light_pos_player_1.sum((1, 2))

        
        #TODO: Double check this
        p0_image_features = jnp.stack((unit_mask_player_0,
                                        unit_mask_player_1,
                                        factory_mask_player_0,
                                        factory_mask_player_1,
                                        unit_power,
                                        unit_ice,
                                        unit_ore,
                                        unit_water,
                                        unit_metal,
                                        factory_power,
                                        factory_ice,
                                        factory_ore,
                                        factory_water,
                                        factory_metal,
                                        board_rubble,
                                        board_ore,
                                        board_ice,
                                        board_lichen,
                                        player_0_lichen_mask,
                                        player_1_lichen_mask,
                                        heavy_pos_player_0,
                                        heavy_pos_player_1,
                                        light_pos_player_0,
                                        light_pos_player_1), 
                                        axis = 1
                                        )
        
        p1_image_features = jnp.stack((unit_mask_player_1,
                                        unit_mask_player_0,
                                        factory_mask_player_1,
                                        factory_mask_player_0,
                                        unit_power,
                                        unit_ice,
                                        unit_ore,
                                        unit_water,
                                        unit_metal,
                                        factory_power,
                                        factory_ice,
                                        factory_ore,
                                        factory_water,
                                        factory_metal,
                                        board_rubble,
                                        board_ore,
                                        board_ice,
                                        board_lichen,
                                        player_1_lichen_mask,
                                        player_0_lichen_mask,
                                        heavy_pos_player_1,
                                        heavy_pos_player_0,
                                        light_pos_player_1,
                                        light_pos_player_0),
                                        axis = 1
                                        )


        return p0_image_features, p1_image_features, p0_num_light, p0_num_heavy, p1_num_light, p1_num_heavy


    def _global_features(self, state, p0_num_light, p0_num_heavy, p1_num_light, p1_num_heavy):
        player_0_id = 0
        player_1_id = 1

        #All these are common
        day = jnp.sin((2*jnp.pi*state.real_env_steps)/1000)*0.3
        night = jnp.cos((2*jnp.pi*state.real_env_steps)/1000)*0.2
        timestep = state.real_env_steps / 1000
        day_night = state.real_env_steps % 50 < 30
        ice_on_map = jnp.sum(state.board.map.ice, (1, 2))
        ore_on_map = jnp.sum(state.board.map.ore, (1, 2))
        ice_entropy = js.entr(state.board.map.ice).mean((1, 2))
        ore_entropy = js.entr(state.board.map.ore).mean((1, 2))
        rubble_entropy = js.entr(state.board.map.rubble).mean((1, 2))

        #All these must be flipped
        friendly_factories = state.n_factories[:, player_0_id]
        enemy_factories = state.n_factories[:, player_1_id]

        friendly_light = p0_num_light
        friendly_heavy = p0_num_heavy
        enemy_light = p1_num_light
        enemy_heavy = p1_num_heavy
        
        #Player 1 lichen
        player_0_lichen_mask = jnp.isin(
            state.board.lichen_strains, state.factories.team_id[:, player_0_id]
        )
        player_1_lichen_mask = jnp.isin(
            state.board.lichen_strains, state.factories.team_id[:, player_0_id]
        )

        friendly_lichen_amount =  jnp.sum(jnp.where(player_0_lichen_mask, state.board.lichen, 0), (1, 2))
        enemy_lichen_amount =  jnp.sum(jnp.where(player_1_lichen_mask, state.board.lichen, 0), (1, 2))
        

        lichen_distribution = (friendly_lichen_amount-enemy_lichen_amount)/jnp.clip(friendly_lichen_amount+enemy_lichen_amount, a_min = 1, a_max = None)


        p0_global_features = jnp.stack((day, 
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
                            lichen_distribution), 
                            axis = 1
                            )
        
        p1_global_features = jnp.stack((day, 
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
                            lichen_distribution * -1), 
                            axis = 1
                            )

        return p0_global_features, p1_global_features

    

    def reward(self, image_features, global_features, played_id):
        #Cargo times mask. Because we take in the corrected features here, the indexing is the same regardless of player
        own_unit_ice = image_features[:, 6]*image_features[:, 0]
        own_unit_ore = image_features[:, 7]*image_features[:, 0]

        own_factory_ice = image_features[:, 11]*image_features[:, 2]
        own_factory_ore = image_features[:, 12]*image_features[:, 2]

        unit_ice_reward = jnp.clip(own_unit_ice - self.previous_unit_ice[played_id], 0, None).mean((1, 2))
        unit_ore_reward = jnp.clip(own_unit_ore - self.previous_unit_ore[played_id], 0, None).mean((1, 2))
        factory_ice_reward = jnp.clip((own_factory_ice - self.previous_factory_ice[played_id]), 0, None).mean((1, 2))*2
        factory_ore_reward = jnp.clip((own_factory_ore - self.previous_factory_ore[played_id]), 0, None).mean((1, 2))*2
        
        light_reward = global_features[:, 10] - self.previous_light_units[played_id]
        heavy_reward = global_features[:, 11] - self.previous_heavy_units[played_id]
        
        self.previous_unit_ice[played_id] = own_unit_ice
        self.previous_unit_ore[played_id] = own_unit_ore
        self.previous_factory_ice[played_id] = own_factory_ice
        self.previous_factory_ore[played_id] = own_factory_ore
        self.previous_light_units[played_id] = global_features[:, 10]
        self.previous_heavy_units[played_id] = global_features[:, 10]

        return unit_ice_reward+unit_ore_reward+factory_ice_reward+factory_ore_reward+light_reward+heavy_reward


    def process_state(self, state):
        #We get the number of light/heavy so we don't have to do it twice
        p0_image_features, p1_image_features, p0_num_light, p0_num_heavy, p1_num_light, p1_num_heavy = self._image_features(state)
        p0_global_features, p1_global_features = self._global_features(state, p0_num_light, p0_num_heavy, p1_num_light, p1_num_heavy)
        p0_reward = self.reward(p0_image_features, p0_global_features, 0)
        p1_reward = self.reward(p1_image_features, p1_global_features, 1)
        
        
        return (p0_image_features, p0_global_features, p0_reward), (p1_image_features, p1_global_features, p1_reward)
