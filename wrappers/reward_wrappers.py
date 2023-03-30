import gym
import numpy as np


class SimpleRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        obs = self.state.get_obs()
        observations = {}
        for k in self.agents:
            observations[k] = obs
        return reward


class IceRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, config) -> None:
        super().__init__(env)
        self.config = config

        self.units = {}

    def reset(self):
        self.previous_ice = 0
        self.previous_ore = 0
        self.previous_water = 0
        self.previous_metal = 0
        return_val = self.env.reset() #NOTE: We do this here because reset wipes the stats
        self.env.state.stats["player_0"]['rewards'] = {'resource_reward' : 0, 'heavy_unit_reward' : 0}
        self.env.state.stats["player_0"]["total_episodic_reward"] = 0
        return return_val


    def reward(self, rewards):
        # NOTE: Only handles player_0 atm

        if len(self.agents) == 0:  # Game is over
            strain_ids = self.state.teams["player_0"].factory_strains
            agent_lichen_mask = np.isin(
                self.state.board.lichen_strains, strain_ids
            )
            lichen = self.state.board.lichen[agent_lichen_mask].sum()
            # TODO: Check for correctness
            reward = self.config['scaling_win'] if rewards["player_0"] > rewards["player_1"] else -self.config['scaling_win'] if rewards["player_0"] < rewards["player_1"] else 0
            reward += np.tanh(lichen/self.config["lichen_divide_value"])*self.config['scaling_lichen']
            #self.env.state.stats["player_0"]['rewards']['end_of_episode_reward'] += reward
            return reward


        #Resource reward
        gen = self.env.state.stats["player_0"]["generation"]
        ice = sum([val for val in gen["ice"].values()]) #HEAVY, LIGHT
        ore = sum([val for val in gen["ore"].values()]) #HEAVY, LIGHT   
        water = gen["water"]
        metal = gen["metal"]

        resource_reward = (ice-self.previous_ice)*self.config["scaling_ice"] + (ore-self.previous_ore)*self.config["scaling_ore"]+(water-self.previous_water)*self.config["scaling_water_made"] + (metal-self.previous_metal)*self.config["scaling_metal_made"]
        self.previous_ice = ice
        self.previous_ore = ore
        self.previous_water = water
        self.previous_metal = metal


        #Unit reward, only heavies atm
        heavy_unit_reward = 0
        
        for unit_id, action in self.prev_actions["player_0"].items():
                if "factory" in unit_id and action == 1:
                    heavy_unit_reward += 1



        #Normalize and return
        num_factories = self.env.state.board.factories_per_team

        resource_reward /= num_factories
        heavy_unit_reward = (heavy_unit_reward*self.config["heavy_unit_reward"])/num_factories
        
        reward = (
            heavy_unit_reward
            + resource_reward
        )

        self.env.state.stats["player_0"]["total_episodic_reward"] += reward
        self.env.state.stats["player_0"]['rewards']['resource_reward'] += resource_reward
        self.env.state.stats["player_0"]['rewards']['heavy_unit_reward'] += heavy_unit_reward

        return reward