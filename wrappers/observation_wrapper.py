from gym import spaces
import gym
import numpy as np

from actions.actions import UNIT_ACTION_IDXS
from actions.action_masking import unit_action_mask, factory_action_mask

DIM_NAMES = ["Unit mask player 0", "Unit mask player 1", "Factory mask player 0", "Factory mask player 1", "Unit power", "Unit ice", "Unit ore", "Factory power", 
            "Factory ice", "Factory ore", "Factory water", "Factory metal", "Rubble on board", "Ore on board", "Ice on board", "Player 0 lichen", "Player 1 lichen", "Heavy units", "Light units", 
            "Collision mask"]

def unit_can_go_mask(array):
    array_1 = np.pad(array.copy(), (1,1))
    can_go_mask = array_1.copy()

    can_go_mask += np.roll(array_1, 1, axis=0)
    can_go_mask += np.roll(array_1, -1, axis=0)
    can_go_mask += np.roll(array_1, 1, axis=1)
    can_go_mask += np.roll(array_1, -1, axis=1)

    return can_go_mask[1:-1, 1:-1]

if __name__ == "__main__":
    a = np.zeros((7, 7))
    a[3, 3] = 1
    print(unit_can_go_mask(a))


class StateSpaceVol2(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, config) -> None:
        
        self.num_envs = config["parallel_envs"]
        self.map_size = config["map_size"]

        self.config = config
        #TODO: Check the bounds on "Box"
        super().__init__(env)
        #NOTE: could not broadcast errors often stem from here

        self.observation_space = spaces.Dict({
                                                "player_0" : spaces.Dict({
                                                                "features": spaces.Box(0, 1, shape = (29, self.map_size, self.map_size)),
                                                                "unit_mask" : spaces.Box(0, 1, shape=(self.map_size, self.map_size)),
                                                                "factory_mask": spaces.Box(0, 1, shape=(self.map_size, self.map_size)),
                                                                "invalid_unit_action_mask": spaces.Box(0, 1, shape=(self.map_size, self.map_size, UNIT_ACTION_IDXS)),
                                                                "invalid_factory_action_mask": spaces.Box(0, 1, shape=(self.map_size, self.map_size, 4)),
                                                            }),
                                                "player_1" : spaces.Dict({
                                                                "features": spaces.Box(0, 1, shape = (29, self.map_size, self.map_size)),
                                                                "unit_mask" : spaces.Box(0, 1, shape=(self.map_size, self.map_size)),
                                                                "factory_mask": spaces.Box(0, 1, shape=(self.map_size, self.map_size)),
                                                                "invalid_unit_action_mask": spaces.Box(0, 1, shape=(self.map_size, self.map_size, UNIT_ACTION_IDXS)),
                                                                "invalid_factory_action_mask": spaces.Box(0, 1, shape=(self.map_size, self.map_size, 4)),
                                                            })})
    def _image_features(self, state):
            player_0_id = "player_0"
            player_1_id = "player_1"

            state = state[player_0_id]
            

            #NOTE: First channel ALWAYS unit_mask, second channel ALWAYS factory mask
            p0_unit_mask = np.zeros((self.map_size, self.map_size))
            p1_unit_mask = np.zeros((self.map_size, self.map_size))

            p0_factory_mask = np.zeros((self.map_size, self.map_size))
            p1_factory_mask = np.zeros((self.map_size, self.map_size))

            factory_occupancy_mask_player_0 = np.zeros((self.map_size, self.map_size))
            factory_occupancy_mask_player_1 = np.zeros((self.map_size, self.map_size))

            unit_power = np.zeros((self.map_size, self.map_size))
            unit_ice = np.zeros((self.map_size, self.map_size))
            unit_ore = np.zeros((self.map_size, self.map_size))

            factory_power = np.zeros((self.map_size, self.map_size))
            factory_ice = np.zeros((self.map_size, self.map_size))
            factory_ore = np.zeros((self.map_size, self.map_size))
            factory_water = np.zeros((self.map_size, self.map_size))
            factory_metal = np.zeros((self.map_size, self.map_size))

            heavy_pos = np.zeros((self.map_size, self.map_size))
            light_pos = np.zeros((self.map_size, self.map_size))
            


            for player, unit_mask, factory_occupancy_mask, factory_mask in zip([player_0_id, player_1_id],
                                        [p0_unit_mask, p1_unit_mask],
                                        [factory_occupancy_mask_player_0, factory_occupancy_mask_player_1],
                                        [p0_factory_mask, p1_factory_mask]):
                
                factories = state["factories"][player]
                units = state["units"][player]
                

                for _, unit in units.items():
                    x, y = unit["pos"]
                    unit_mask[x, y] = 1
                    is_light = unit["unit_type"] == "LIGHT"
                    if is_light:
                        light_pos[x, y] = 1
                    else:
                        heavy_pos[x, y] = 1
                    unit_power[x, y] = unit["power"]/(500) #NOTE: THIS IS WEIRD
                    unit_ice[x, y] = unit["cargo"]["ice"]/(100 if is_light else 1000)
                    unit_ore[x, y] = unit["cargo"]["ore"]/(100 if is_light else 1000)
                    
            
                for _, factory in factories.items():
                    x, y = factory["pos"]
                    factory_occupancy_mask[x-1:x+2, y-1:y+2] = 1
                    factory_mask[x, y] = 1
                    #Factories have no max capacity
                    #TODO: Look at tanh?
                    factory_power[x-1:x+2, y-1:y+2] = min(1, factory["power"]/500)              
                    factory_ice[x-1:x+2, y-1:y+2] = min(1, factory["cargo"]["ice"]/200)
                    factory_ore[x-1:x+2, y-1:y+2] = min(1, factory["cargo"]["ore"]/200)
                    factory_water[x-1:x+2, y-1:y+2] = min(1, factory["cargo"]["water"]/200)
                    factory_metal[x-1:x+2, y-1:y+2] = min(1, factory["cargo"]["metal"]/200)

            board_rubble = state["board"]["rubble"]/self.env.state.env_cfg.MAX_RUBBLE
            board_ice = state["board"]["ice"]
            board_ore = state["board"]["ore"]
            board_lichen = state["board"]["lichen"]/self.env.state.env_cfg.MAX_LICHEN_PER_TILE
            
            player_0_lichen_mask = np.isin(
                self.env.state.board.lichen_strains, self.state.teams["player_0"].factory_strains
            )
            player_1_lichen_mask = np.isin(
                self.env.state.board.lichen_strains, self.state.teams["player_1"].factory_strains
            )

            collision_mask = unit_can_go_mask(p0_unit_mask + p1_unit_mask) > 1
            
            #TODO: Double check this
            p0_image_features = np.stack((      p0_unit_mask,
                                                p1_unit_mask,
                                                factory_occupancy_mask_player_0,
                                                factory_occupancy_mask_player_1,
                                                unit_power,
                                                unit_ice,
                                                unit_ore,
                                                factory_power,
                                                factory_ice,
                                                factory_ore,
                                                factory_water,
                                                factory_metal,
                                                board_rubble,
                                                board_ore,
                                                board_ice,
                                                player_0_lichen_mask*board_lichen,
                                                player_1_lichen_mask*board_lichen,
                                                heavy_pos,
                                                light_pos,
                                                collision_mask), 
                                                axis = 0
                                                )
            
            p1_image_features = np.stack((      p1_unit_mask,
                                                p0_unit_mask,
                                                factory_occupancy_mask_player_1,
                                                factory_occupancy_mask_player_0,
                                                unit_power,
                                                unit_ice,
                                                unit_ore,
                                                factory_power,
                                                factory_ice,
                                                factory_ore,
                                                factory_water,
                                                factory_metal,
                                                board_rubble,
                                                board_ore,
                                                board_ice,
                                                player_1_lichen_mask*board_lichen,
                                                player_0_lichen_mask*board_lichen,
                                                heavy_pos,
                                                light_pos,
                                                collision_mask),
                                                axis = 0
                                                )

            return p0_image_features, p0_unit_mask, p0_factory_mask, p1_image_features, p1_unit_mask, p1_factory_mask

    def _global_features(self, state):
        #TODO: Fill like, and stack with image
        player_0_id = "player_0"
        player_1_id = "player_1"

        #All these are common
        day = np.ones((self.map_size, self.map_size))*np.sin((2*np.pi*self.env.state.real_env_steps)/1000)*0.3
        night = np.ones((self.map_size, self.map_size))*np.cos((2*np.pi*self.env.state.real_env_steps)/1000)*0.2
        timestep = np.ones((self.map_size, self.map_size))*self.env.state.real_env_steps / 1000
        day_night = np.ones((self.map_size, self.map_size))*self.env.state.real_env_steps % 50 < 30
        ice_on_map = np.ones((self.map_size, self.map_size))*np.sum(state[player_0_id]["board"]["ice"]) / 30
        ore_on_map = np.ones((self.map_size, self.map_size))*np.sum(state[player_0_id]["board"]["ore"]) / 30

        #All these must be flipped
        friendly_factories = np.ones((self.map_size, self.map_size))*len(state[player_0_id]["factories"][player_0_id].values()) /4
        enemy_factories = np.ones((self.map_size, self.map_size))*len(state[player_0_id]["factories"][player_1_id].values()) /4

        #friendly_light = p0_num_light
        #friendly_heavy = p0_num_heavy
        #enemy_light = p1_num_light
        #enemy_heavy = p1_num_heavy
        
        #Player 1 lichen
        player_0_lichen_mask = np.isin(
            self.env.state.board.lichen_strains, self.state.teams[player_0_id].factory_strains
        )
        player_1_lichen_mask = np.isin(
            self.env.state.board.lichen_strains, self.state.teams[player_1_id].factory_strains
        )

        friendly_lichen_amount =  np.ones((self.map_size, self.map_size))*np.sum(np.where(player_0_lichen_mask, self.env.state.board.lichen, 0))
        enemy_lichen_amount =  np.ones((self.map_size, self.map_size))*np.sum(np.where(player_1_lichen_mask, self.env.state.board.lichen, 0))
        

        lichen_distribution = np.ones((self.map_size, self.map_size))*(friendly_lichen_amount-enemy_lichen_amount)/np.clip(friendly_lichen_amount+enemy_lichen_amount, a_min = 1, a_max = None)


        p0_global_features = np.stack((     day, 
                                            night, 
                                            timestep, 
                                            day_night, 
                                            ice_on_map, 
                                            ore_on_map, 
                                            friendly_factories, 
                                            #friendly_light, 
                                            #friendly_heavy, 
                                            enemy_factories, 
                                            #enemy_light, 
                                            #enemy_heavy, 
                                            lichen_distribution), 
                                            axis = 0
                                            )
        
        p1_global_features = np.stack(( day, 
                                        night, 
                                        timestep, 
                                        day_night, 
                                        ice_on_map, 
                                        ore_on_map,
                                        enemy_factories, 
                                        #enemy_light, 
                                        #enemy_heavy, 
                                        friendly_factories, 
                                        #friendly_light, 
                                        #friendly_heavy, 
                                        lichen_distribution * -1), 
                                        axis = 0
                                        )

        return p0_global_features.astype(np.float32), p1_global_features.astype(np.float32)

    def process_state(self, obs):
        #We get the number of light/heavy so we don't have to do it twice
        p0_image_features, p0_unit_mask, p0_factory_mask, p1_image_features, p1_unit_mask, p1_factory_mask = self._image_features(obs)
        p0_global_features, p1_global_features = self._global_features(obs)
        p0_features = np.concatenate((p0_image_features, p0_global_features))
        p1_features = np.concatenate((p1_image_features, p1_global_features))
        return p0_features, p0_unit_mask, p0_factory_mask, p1_features, p1_unit_mask, p1_factory_mask
    

    def observation(self, obs):
        p0_features, p0_unit_mask, p0_factory_mask, p1_features, p1_unit_mask, p1_factory_mask = self.process_state(obs)
        self.last_obs = obs #TODO: I hope these don't get changed?
        self.last_state_p0 = p0_features 
        self.last_state_p1 = p1_features

        p0_features = {"features": p0_features.astype(np.float32),
                       "unit_mask": p0_unit_mask.astype(np.float32), 
                       "factory_mask": p0_factory_mask.astype(np.float32),
                       "invalid_unit_action_mask": unit_action_mask(obs, p0_features, "player_0").astype(np.float32),
                       "invalid_factory_action_mask": factory_action_mask(obs, "player_0").astype(np.float32)
                       }
        
        p1_features = {"features": p1_features.astype(np.float32),
                       "unit_mask": p1_unit_mask.astype(np.float32), 
                       "factory_mask": p1_factory_mask.astype(np.float32),
                       "invalid_unit_action_mask": unit_action_mask(obs, p1_features, "player_1").astype(np.float32),
                       "invalid_factory_action_mask": factory_action_mask(obs, "player_1").astype(np.float32)
                       }
        
        return {"player_0": p0_features, "player_1": p1_features}