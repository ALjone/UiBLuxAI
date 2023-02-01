from typing import Dict

import torch
import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

class StateSpaceVol1(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)


    def observation(self, obs):
        shared_obs = obs["player_0"]

        main_player = self.agents[0]
        #NOTE: First channel ALWAYS unit_mask, second channel ALWAYS factory mask
        image_features = torch.zeros((33, 48, 48))
        global_features = torch.zeros(15)

        unit_mask = torch.zeros((48, 48))
        factory_mask = torch.zeros((48, 48))

        unit_cargo = torch.zeros((3, 48, 48)) #Power, Ice, Ore
        factory_cargo = torch.zeros((5, 48, 48))

        board = torch.zeros((4, 48, 48)) #Rubble, Ice, Ore, Lichen

        lichen_mask = torch.zeros((48, 48)) #1 for friendly, 0 for none, -1 for enemy

        unit_type = torch.zeros(2, 48, 48) #LIGHT, HEAVY

        action_queue_length = torch.zeros((48, 48))
        #TODO: Move this to agent
        action_queue_type_friendly = torch.zeros((14, 48, 48))

        next_step = torch.zeros((2, 48, 48))

        day = 0
        night = 0
        timestep = 0
        lichen_distribution = 0
        day_night = 0
        friendly_factories = 0
        enemy_factories = 0

        friendly_light = 0
        friendly_heavy = 0
        enemy_light = 0
        enemy_heavy = 0
        ice_on_map = 0
        ore_on_map = 0
        ice_entropy = 0
        ore_entropy = 0
        rubble_entropy = 0
        
        for player in self.agents:
            factories = shared_obs["factories"][player]
            units = shared_obs["units"][player]

            for _, unit in units.items():
                x, y = unit["pos"]
                unit_mask[x, y] = (1 if player == main_player else -1)
                is_light = unit["unit_type"] == "LIGHT"
                if is_light:
                    unit_type[0, x, y] = 1
                else:
                    unit_type[1, x, y] = 1
                unit_cargo[0, x, y] = unit["power"]/(150 if is_light else 3000)
                unit_cargo[1, x, y] = unit["cargo"]["ice"]/(100 if is_light else 1000)
                unit_cargo[2, x, y] = unit["cargo"]["ore"]/(100 if is_light else 1000)

                print(unit.keys())
                pass
        
            for _, factory in factories.items():
                x, y = factory["pos"]
                factory_mask[x, y] = (1 if player == main_player else -1)
                #TODO: Look at tanh?
                factory_cargo[0, x, y] = factory["power"]/1000              
                factory_cargo[1, x, y] = factory["cargo"]["ice"]/1000
                factory_cargo[2, x, y] = factory["cargo"]["ore"]/1000
                factory_cargo[3, x, y] = factory["cargo"]["water"]/1000
                factory_cargo[4, x, y] = factory["cargo"]["metal"]/1000

        board[0] = shared_obs["board"]["rubble"]
        board[1] = shared_obs["board"]["ice"]
        board[2] = shared_obs["board"]["ore"]
        board[3] = shared_obs["board"]["lichen"]
        #TODO: Flip like Fillip
        image_state = {}

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