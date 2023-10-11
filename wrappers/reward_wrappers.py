import gym
import numpy as np

class IceRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, config) -> None:
        super().__init__(env)
        self.config = config

        self.units = {}
        self.player = "player_0"
        self.opponent = "player_1"

    def reset(self):
        self.previous_ice = 0
        self.previous_ore = 0
        self.previous_ice_deposited = 0
        self.previous_ore_deposited = 0

        self.previous_rubble_destroyed = 0 #self.env.state.get_obs()["board"]["rubble"]
        self.previous_destroyed_factories = 0
        self.previous_light_count = 0
        self.previous_heavy_count = 0

        return_val = self.env.reset() #NOTE: We do this here because reset wipes the stats
        self.env.state.stats["player_0"]['rewards'] = {'resource_reward' : 0, 'unit_reward' : 0,
                                                        'rubble_reward': 0, "factory_punishment" : 0,
                                                          "win_reward": 0, "env_reward": 0}
        self.env.state.stats["player_0"]["total_episodic_reward"] = 0
        return return_val
    
    def get_reward_scaling(self):
        games_played = self.games_played
        games_to_anneal_over = (self.config["total_timesteps"] / (1000*self.config["parallel_envs"]))

        # Calculate the reward scaling factor based on the number of games played
        reward_scaling = 1 - (games_played / games_to_anneal_over)
        
        return max(0, reward_scaling)
    
    def phase_1(self):

        stats = self.env.state.stats[self.player]
        rubble_destroyed = (sum([val for val in stats["destroyed"]["rubble"].values()])-self.previous_rubble_destroyed)

        #Resource reward
        #ice_deposited, ore_deposited, ice_mined, ore_mined = self.transfer_reward()
        ice_mined = sum([val for val in stats["generation"]["ice"].values()]) - self.previous_ice
        ore_mined = sum([val for val in stats["generation"]["ore"].values()]) - self.previous_ore

        #TODO: Only works as long as you can _only_ transfer to factory
        ice_deposited = stats["transfer"]["ice"] - self.previous_ice_deposited
        ore_deposited = stats["transfer"]["ore"] - self.previous_ore_deposited

        resource_reward = (ice_mined*self.config["scaling_ice"] 
                        + ore_mined*self.config["scaling_ore"]
                        + ice_deposited*self.config["scaling_water"] 
                        + ore_deposited*self.config["scaling_metal"])

        #Normalize and return
        num_factories = self.env.state.board.factories_per_team

        resource_reward /= num_factories
        rubble_reward = (rubble_destroyed*self.config["rubble_reward"])/num_factories
        
        
        light_count = stats["generation"]["built"]["LIGHT"]-stats["destroyed"]["LIGHT"]
        heavy_count = stats["generation"]["built"]["HEAVY"]-stats["destroyed"]["HEAVY"]

        #light_count = 0
        #heavy_count = 0
        #for unit in self.state.units[player].values():
        #    light_count += 1 if unit.unit_type == "LIGHT" else 0
        #    heavy_count += 1 if unit.unit_type == "HEAVY" else 0
        #light_units_reward = (light_count-self.previous_light_count)*self.config["light_reward"]
        #heavy_units_reward = (heavy_count-self.previous_heavy_count)*self.config["heavy_reward"]
        light_units_reward = light_count*self.config["light_reward"]/num_factories
        heavy_units_reward = heavy_count*self.config["heavy_reward"]/num_factories

        scaling = self.config["reward_scale_start"]
        if self.config["anneal_rewards"]:
            scaling *= self.get_reward_scaling()
        resource_reward *= scaling
        light_units_reward*= scaling
        heavy_units_reward*= scaling
        rubble_reward*=scaling
            

        reward = (
            resource_reward
            + light_units_reward
            + heavy_units_reward
            + rubble_reward
        )


        self.previous_light_count = light_count
        self.previous_heavy_count = heavy_count
        self.previous_ice = sum([val for val in stats["generation"]["ice"].values()])
        self.previous_ore = sum([val for val in stats["generation"]["ore"].values()])
        self.previous_ice_deposited = stats["transfer"]["ice"]
        self.previous_ore_deposited = stats["transfer"]["ore"]
        self.previous_rubble_destroyed = sum([val for val in stats["destroyed"]["rubble"].values()])

        stats["total_episodic_reward"] += reward
        stats['rewards']['resource_reward'] += resource_reward
        stats['rewards']['rubble_reward'] += rubble_reward
        stats["rewards"]["unit_reward"] += light_units_reward + heavy_units_reward

        return reward
    
    def phase_3(self, rewards):
        stats = self.env.state.stats[self.player]
        win_reward = 0
        if self.env.state.real_env_steps == 1000 or rewards[self.opponent] == -1000 or rewards[self.player] == -1000:
            if rewards[self.opponent] > rewards[self.player]:
                win_reward = -1
            elif rewards[self.opponent] < rewards[self.player]:
                win_reward = 1

        stats["rewards"]["win_reward"] += win_reward

        stats["total_episodic_reward"] += win_reward
        return win_reward
        
    def phase_2(self, rewards):
        stats = self.env.state.stats[self.player]

        env_reward = 0
        if self.env.state.real_env_steps == 1000: #Only reward lichen if end of game
            env_reward += (rewards[self.player]-rewards[self.opponent])/np.clip(rewards[self.player]+rewards[self.opponent], a_min = 1, a_max = None)
        
        #Factories lost reward
        factories_lost_reward = -(stats["destroyed"]["FACTORY"]-self.previous_destroyed_factories)*self.config["factory_lost"]
        self.previous_destroyed_factories = stats["destroyed"]["FACTORY"]

        if self.config["anneal_rewards"]:
            scaling = self.get_reward_scaling()
            factories_lost_reward *= scaling
            env_reward *= scaling

        stats["rewards"]["factory_punishment"] += factories_lost_reward
        stats["rewards"]["env_reward"] += env_reward

        stats["total_episodic_reward"] += factories_lost_reward + env_reward
        return factories_lost_reward + env_reward

    def reward(self, rewards):
        # NOTE: Only handles player_0 atm

        reward = 0
        if 3 in self.config["phases"]:
            reward += self.phase_3(rewards)
        
        if 2 in self.config["phases"]:
            reward += self.phase_2(rewards)
        
        if 1 in self.config["phases"]:
            reward += self.phase_1()


        return reward