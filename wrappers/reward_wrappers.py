import gym
import numpy as np

class IceRewardWrapper(gym.RewardWrapper):

    def __init__(self, env, config) -> None:
        super().__init__(env)
        self.config = config

        self.units = {}

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
        self.env.state.stats["player_0"]['rewards'] = {'resource_reward' : 0, 'unit_reward' : 0, 'rubble_reward': 0, "factory_punishment" : 0, "end_of_game_reward": 0}
        self.env.state.stats["player_0"]["total_episodic_reward"] = 0
        return return_val
    

    def reward(self, rewards):
        # NOTE: Only handles player_0 atm

        """if self.env.state.real_env_steps == 1000:  # Game is over
            strain_ids = self.state.teams["player_0"].factory_strains
            agent_lichen_mask = np.isin(
                self.state.board.lichen_strains, strain_ids
            )
            lichen = self.state.board.lichen[agent_lichen_mask].sum()
            # TODO: Check for correctness
            reward = self.config['scaling_win'] if rewards["player_0"] > rewards["player_1"] else -self.config['scaling_win'] if rewards["player_0"] < rewards["player_1"] else 0
            reward += np.tanh(lichen/self.config["lichen_divide_value"])*self.config['scaling_lichen']
            #stats['rewards']['end_of_episode_reward'] += reward
            return reward"""
    
        stats = self.env.state.stats["player_0"]

        #Factories lost reward
        factories_lost_reward = -(stats["destroyed"]["FACTORY"]-self.previous_destroyed_factories)*self.config["factory_lost"]
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
        
        end_of_game_reward = 0

        if self.env.state.real_env_steps == 1000:
            player_0_lichen_mask = np.isin(
                self.env.state.board.lichen_strains, self.state.teams["player_0"].factory_strains
            )
            player_1_lichen_mask = np.isin(
                self.env.state.board.lichen_strains, self.state.teams["player_1"].factory_strains
            )

            friendly_lichen_amount =  np.sum(np.where(player_0_lichen_mask, self.env.state.board.lichen, 0))
            enemy_lichen_amount =  np.sum(np.where(player_1_lichen_mask, self.env.state.board.lichen, 0))
        

            lichen_distribution = (friendly_lichen_amount-enemy_lichen_amount)/np.clip(friendly_lichen_amount+enemy_lichen_amount, a_min = 1, a_max = None)
            end_of_game_reward += self.config["scaling_win"]*lichen_distribution + self.config["end_of_game_reward"]
        
        light_count = stats["generation"]["built"]["LIGHT"]-stats["destroyed"]["LIGHT"]
        heavy_count = stats["generation"]["built"]["HEAVY"]-stats["destroyed"]["HEAVY"]
        light_units_reward = (light_count-self.previous_light_count)*self.config["light_reward"]
        heavy_units_reward = (heavy_count-self.previous_heavy_count)*self.config["heavy_reward"]

        reward = (
            resource_reward
            + light_units_reward
            + heavy_units_reward
            + rubble_reward
            + factories_lost_reward
            + end_of_game_reward
        )


        self.previous_light_count = light_count
        self.previous_heavy_count = heavy_count
        self.previous_ice = sum([val for val in stats["generation"]["ice"].values()])
        self.previous_ore = sum([val for val in stats["generation"]["ore"].values()])
        self.previous_ice_deposited = stats["transfer"]["ice"]
        self.previous_ore_deposited = stats["transfer"]["ore"]
        self.previous_destroyed_factories = stats["destroyed"]["FACTORY"]
        self.previous_rubble_destroyed = sum([val for val in stats["destroyed"]["rubble"].values()])

        stats["total_episodic_reward"] += reward
        stats['rewards']['resource_reward'] += resource_reward
        stats['rewards']['rubble_reward'] += rubble_reward
        stats["rewards"]["factory_punishment"] += factories_lost_reward
        stats["rewards"]["end_of_game_reward"] += end_of_game_reward
        stats["rewards"]["unit_reward"] += light_units_reward + heavy_units_reward

        return reward
    
