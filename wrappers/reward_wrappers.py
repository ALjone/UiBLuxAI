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
        self.previous_units = 0
        self.previous_rubble = self.env.state.get_obs()["board"]["rubble"]
        return_val = self.env.reset() #NOTE: We do this here because reset wipes the stats
        self.env.state.stats["player_0"]['rewards'] = {'resource_reward' : 0, 'unit_punishment' : 0, 'rubble_reward': 0, "factory_punishment": 0}
        self.env.state.stats["player_0"]["total_episodic_reward"] = 0
        return return_val
    

    def unit_rewards(self):
        obs = self.env.state.get_obs()
        rubble = obs["board"]["rubble"]
        rubble_mined = 0

        units = len(obs["units"]["player_0"].keys())
        units_made = 0
        for id, action in self.prev_actions["player_0"].items():
            if action in [0, 1]: #Unit was generated
                units_made += 1
            if isinstance(action, list):
                if action[0][0] == 3 and id in obs["units"]["player_0"].keys():
                    x, y = obs["units"]["player_0"][id]["pos"]
                    rubble_mined += max(self.previous_rubble[x, y] - rubble[x, y], 0) #Max of this and 0 because dying factories leave behind rubble, although this should be masked away

        unit_reward = units - (self.previous_units+units_made) #How many units you should have
        self.previous_units = units
        self.previous_rubble = rubble
        return unit_reward, rubble_mined

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
        ice = gen["ice"]["HEAVY"] #HEAVY, LIGHT
        ore = gen["ore"]["HEAVY"] #HEAVY, LIGHT   
        water = gen["water"]
        metal = gen["metal"]

        resource_reward = (ice-self.previous_ice)*self.config["scaling_ice"] + (ore-self.previous_ore)*self.config["scaling_ore"]+(water-self.previous_water)*self.config["scaling_water_made"] + (metal-self.previous_metal)*self.config["scaling_metal_made"]
        self.previous_ice = ice
        self.previous_ore = ore
        self.previous_water = water
        self.previous_metal = metal

        #Unit reward, only heavies atm
        unit_punishment, rubble_reward = self.unit_rewards()

        #Normalize and return
        num_factories = self.env.state.board.factories_per_team

        resource_reward /= num_factories
        rubble_reward = (rubble_reward*self.config["rubble_reward"])/num_factories
        unit_punishment = (unit_punishment*self.config["unit_punishment"])/num_factories
        
        reward = (
            unit_punishment
            + resource_reward
            + rubble_reward
        )

        self.env.state.stats["player_0"]["total_episodic_reward"] += reward
        self.env.state.stats["player_0"]['rewards']['resource_reward'] += resource_reward
        self.env.state.stats["player_0"]['rewards']['unit_punishment'] += unit_punishment
        self.env.state.stats["player_0"]['rewards']['rubble_reward'] += rubble_reward

        return reward