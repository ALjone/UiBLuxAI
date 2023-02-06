from typing import Dict

import torch
import gym
import numpy as np
import numpy.typing as npt
from gym import spaces


#Delta change, idx to mapping
#TODO: Triple check this!!!
dirs = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]

class StateSpaceVol1(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def image_features(self, obs):
        main_player = self.agents[0]
        shared_obs = obs[main_player]

        #NOTE: First channel ALWAYS unit_mask, second channel ALWAYS factory mask

        unit_mask = np.zeros((1, 48, 48))
        #TODO: We also need a map for where factories are occupying space? Since they are 3x3
        factory_mask = np.zeros((1, 48, 48))

        unit_cargo = np.zeros((3, 48, 48)) #Power, Ice, Ore
        factory_cargo = np.zeros((5, 48, 48))

        board = np.zeros((4, 48, 48)) #Rubble, Ice, Ore, Lichen

        lichen_mask = np.zeros((1, 48, 48)) #1 for friendly, 0 for none, -1 for enemy

        unit_type = np.zeros((2, 48, 48)) #LIGHT, HEAVY

        #TODO: Move this to agent
        action_queue_length = np.zeros((1, 48, 48))

        next_step = np.zeros((2, 48, 48))
        
        for i, player in enumerate(self.agents):
            factories = shared_obs["factories"][player]
            units = shared_obs["units"][player]

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
            if player in self.state.teams.keys():
                strain_ids = self.state.teams[player].factory_strains
                agent_lichen_mask = np.isin(
                    self.state.board.lichen_strains, strain_ids
                )
                lichen_mask += agent_lichen_mask * (1 if player == main_player else -1)

        board[0] = shared_obs["board"]["rubble"]/self.env.state.env_cfg.MAX_RUBBLE
        board[1] = shared_obs["board"]["ice"]
        board[2] = shared_obs["board"]["ore"]
        board[3] = shared_obs["board"]["lichen"]/self.env.state.env_cfg.MAX_LICHEN_PER_TILE
        
        #TODO: Add action queue type in RL agent
        #Don't ask why this is np to torch...
        image_features = torch.tensor(np.concatenate([
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

        return image_features, image_features_flipped
    
    def global_features(self, obs):
        main_player = self.agents[0]
        other_player = self.agents[1]
        shared_obs = obs[main_player]


        #All these are common
        day = np.sin((2*np.pi*self.env.state.real_env_steps)/1000)*0.3
        night = np.cos((2*np.pi*self.env.state.real_env_steps)/1000)*0.2
        timestep = self.env.state.real_env_steps / 1000
        day_night = (1 if self.env.state.real_env_steps % 50 < 30 else 0)
        ice_on_map = np.sum(shared_obs["board"]["ice"])
        ore_on_map = np.sum(shared_obs["board"]["ore"])
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
        if main_player in self.state.teams.keys():
            strain_ids = self.state.teams[main_player].factory_strains
            agent_lichen_mask = np.isin(
                self.state.board.lichen_strains, strain_ids
            )
            friendly_lichen_amount =  np.sum(shared_obs["board"]["lichen"]*agent_lichen_mask)

            strain_ids = self.state.teams[other_player].factory_strains
            agent_lichen_mask = np.isin(
                self.state.board.lichen_strains, strain_ids
            )
            enemy_lichen_amount =  np.sum(shared_obs["board"]["lichen"]*agent_lichen_mask)
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


        return main_player_vars, other_player_vars


    def observation(self, obs):
        #If we're still in early phase
        if len(self.agents) == 0:
            return "", obs
        main_player = self.agents[0]
        other_player = self.agents[1]

        
        main_player_image_features, other_player_image_features = self.image_features(obs)

        main_player_global_features, other_player_global_features = self.global_features(obs)

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

class ImageWithUnitsWrapper(gym.ObservationWrapper):

    """Wrapper, based on the one in the Lux AI Kit, that also returns a mapping from pos -> unit id, so that actions can actually be done"""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_dims = 23  # see _convert_obs function for how this is computed
        self.map_size = self.env.env_cfg.map_size
        self.observation_space = spaces.Box(
            -999, 999, shape=(self.map_size, self.map_size, obs_dims)
        )

    def observation(
        self, obs) -> Dict[str, npt.NDArray]:   
        """Returns the image features as a torch tensor"""
        shared_obs = obs["player_0"]

        unit_mask = np.zeros((self.map_size, self.map_size, 1))
        unit_data = np.zeros(
            (self.map_size, self.map_size, 9)
        )  # power(1) + cargo(4) + unit_type(1) + unit_pos(2) + team(1)
        factory_mask = np.zeros_like(unit_mask)
        factory_data = np.zeros(
            (self.map_size, self.map_size, 8)
        )  # power(1) + cargo(4) + factory_pos(2) + team(1)
        for agent in ["player_0"]:
            factories = shared_obs["factories"][agent]
            units = shared_obs["units"][agent]

            for unit_id in units.keys():
                unit = units[unit_id]
                # we encode everything but unit_id or action queue
                cargo_space = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].CARGO_SPACE
                battery_cap = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                unit_vec = np.concatenate(
                    [unit["pos"], [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )
                unit_vec[:2] /= self.env.state.env_cfg.map_size

                # note that all data is stored as map[x, y] format
                unit_data[unit["pos"][0], unit["pos"][1]] = unit_vec
                unit_mask[unit["pos"][0], unit["pos"][1]] = 1

            for unit_id in factories.keys():
                factory = factories[unit_id]
                # we encode everything but strain_id or unit_id
                cargo_vec = np.array(
                    [
                        factory["power"],
                        factory["cargo"]["ice"],
                        factory["cargo"]["ore"],
                        factory["cargo"]["water"],
                        factory["cargo"]["metal"],
                    ]
                )
                cargo_vec = cargo_vec * 1 / 1000

                factory_vec = np.concatenate(
                    [factory["pos"], cargo_vec, [factory["team_id"]]], axis=-1
                )
                factory_vec[:2] /= self.env.state.env_cfg.map_size
                factory_data[factory["pos"][0], factory["pos"][1]] = factory_vec
                factory_mask[factory["pos"][0], factory["pos"][1]] = 1 if factory["team_id"] == 0 else 0

            #NOTE: Unit mask MUST be first, factory mask MUST be second
            image_features = np.concatenate(
                [
                    unit_mask,
                    factory_mask,
                    factory_data,
                    np.expand_dims(shared_obs["board"]["lichen"], -1)
                    / self.env.state.env_cfg.MAX_LICHEN_PER_TILE,
                    np.expand_dims(shared_obs["board"]["rubble"], -1)
                    / self.env.state.env_cfg.MAX_RUBBLE,
                    np.expand_dims(shared_obs["board"]["ice"], -1),
                    np.expand_dims(shared_obs["board"]["ore"], -1),
                    unit_data,
                ],
                axis=-1,
            )

        image_features = torch.from_numpy(image_features.transpose(2, 0, 1))

        new_obs = dict()
        #NOTE: This is hardcoded, maybe not so smart, but since self.agents is emptied at the end of the game, it has to be done
        for agent in ["player_0", "player_1"]:
            new_obs[agent] = {}
            new_obs[agent]["image_features"] = image_features.type(torch.float32) #TODO Shouldn't this dependet on agent?

            new_obs[agent]["unit_to_id"] = shared_obs["units"][agent].values()
            new_obs[agent]["factory_to_id"] = shared_obs["factories"][agent].values()

        return new_obs, obs