from typing import Dict

import copy
import torch
import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

from lux.kit import obs_to_game_state
from lux.utils import my_turn_to_place_factory

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
        for agent in obs.keys():
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
                factory_mask[factory["pos"][0], factory["pos"][1]] = 1

            image_features = np.concatenate(
                [
                    np.expand_dims(shared_obs["board"]["lichen"], -1)
                    / self.env.state.env_cfg.MAX_LICHEN_PER_TILE,
                    np.expand_dims(shared_obs["board"]["rubble"], -1)
                    / self.env.state.env_cfg.MAX_RUBBLE,
                    np.expand_dims(shared_obs["board"]["ice"], -1),
                    np.expand_dims(shared_obs["board"]["ore"], -1),
                    unit_mask,
                    unit_data,
                    factory_mask,
                    factory_data,
                ],
                axis=-1,
            )

        image_features = torch.from_numpy(image_features.transpose(2, 0, 1))

        new_obs = dict()
        #NOTE: This is hardcoded, maybe not so smart, but since self.agents is emptied at the end of the game, it has to be done
        for agent in ["player_0", "player_1"]:
            new_obs[agent] = {}
            new_obs[agent]["image_features"] = image_features #TODO Shouldn't this dependet on agent?

            new_obs[agent]["unit_to_id"] = shared_obs["units"][agent].values()
            new_obs[agent]["factory_to_id"] = shared_obs["factories"][agent].values()

        return new_obs, obs

def get_factory(step, env_cfg, obs, player):
    if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
    else:
        game_state = obs_to_game_state(step, env_cfg, obs)
        # factory placement period

        # how many factories you have left to place
        factories_to_place = game_state.teams[player].factories_to_place
        # whether it is your turn to place a factory
        my_turn_to_place = my_turn_to_place_factory(game_state.teams[player].place_first, step)
        if factories_to_place > 0 and my_turn_to_place:
            # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
            potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
            spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
            return dict(spawn=spawn_loc, metal=150, water=150)
        return dict()

class SinglePlayerEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = None


    def get_reward(self, action, obs):
        #NOTE: Copied from the kit
        agent = "player_0"
        shared_obs = obs["player_0"]

        # compute reward
        # we simply want to encourage the heavy units to move to ice tiles
        # and mine them and then bring them back to the factory and dump it
        # as well as survive as long as possible

        factories = shared_obs["factories"][agent]
        factory_pos = None
        for unit_id in factories:
            factory = factories[unit_id]
            # note that ice converts to water at a 4:1 ratio
            factory_pos = np.array(factory["pos"])
            break
        units = shared_obs["units"][agent]
        unit_deliver_ice_reward = 0
        unit_move_to_ice_reward = 0
        unit_overmining_penalty = 0
        penalize_power_waste = 0

        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        unit_power = 0
        for unit_id in units:
            unit = units[unit_id]
            if unit["unit_type"] == "HEAVY":
                pos = np.array(unit["pos"])
                ice_tile_distances = np.mean((ice_tile_locations - pos) ** 2, 1)
                closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                dist_to_ice = manhattan_dist(closest_ice_tile, pos)
                unit_power = unit["power"]
                if unit["cargo"]["ice"] < 20:

                    dist_penalty = min(
                        1.0, dist_to_ice / (10)
                    )  # go beyond 12 squares manhattan dist and no reward
                    unit_move_to_ice_reward += (
                        1 - dist_penalty
                    ) * 0.1  # encourage unit to move to ice
                else:
                    if factory_pos is not None:
                        dist_to_factory = manhattan_dist(pos, factory_pos)
                        dist_penalty = min(1.0, dist_to_factory / 10)
                        unit_deliver_ice_reward = (
                            0.2 + (1 - dist_penalty) * 0.1
                        )  # encourage unit to move back to factory


        #print(self.state.stats["player_0"]["lichen_grown"].keys())
        #print(self.state.stats["player_0"]["units_built"]["LIGHT"])
        reward = (
            0
            + unit_move_to_ice_reward
            + unit_deliver_ice_reward
            + unit_overmining_penalty
            #+ (self.state.stats["player_0"]["units_built"]["LIGHT"]/(len(self.env.state.factories[agent])*15)) if len(self.agents) > 0 else 0
        )
        reward = reward


        return {"player_0" : reward}

    def step(self, action):
        agent = self.agents[0]
        opp_agent = self.agents[1]

        opp_factories = self.env.state.factories[opp_agent]

        for k in opp_factories:
            factory = opp_factories[k]
            factory.cargo.water = 1000 # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent

        if self.state.real_env_steps < 0:
            opp_action = get_factory(self.state.env_steps, self.env.state.env_cfg, self.state.get_obs(), opp_agent)
        else:
            opp_action = {}

        obs, reward, done, info = super().step({agent: action, opp_agent: opp_action})
         
        if self.state.real_env_steps < 1:
            reward = {agent : 0}
        else:
            reward = self.get_reward(action, obs[1]) 

        #NOTE: See here for tips on custom reward function https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/examples/sb3.py
        return (obs[0][agent], obs[1]), reward[agent], done[agent], info[agent]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        return (obs[0][self.agents[0]], obs[1])