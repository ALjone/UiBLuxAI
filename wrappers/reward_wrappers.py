import gym

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
        self.previous_rubble = self.env.state.get_obs()["board"]["rubble"]
        self.previous_destroyed_factories = 0
        self.previous_units = {}
        return_val = self.env.reset() #NOTE: We do this here because reset wipes the stats
        self.env.state.stats["player_0"]['rewards'] = {'resource_reward' : 0, 'unit_punishment' : 0, 'rubble_reward': 0, "factory_punishment" : 0}
        self.env.state.stats["player_0"]["total_episodic_reward"] = 0
        return return_val
    

    def transfer_reward(self):
        obs = self.env.state.get_obs()
        units = obs["units"]["player_0"]

        ice_deposited = 0
        ore_deposited = 0
        ice_mined = 0
        ore_mined = 0
        #NOTE: Only way for a unit to lose resources is if it transferes to a factory. Could this be done by simply looking at the stats? 
        for id, unit in units.items():
            if id in self.previous_units.keys():
                ice_deposited += max(0, self.previous_units[id]["cargo"]["ice"]-unit["cargo"]["ice"])
                ore_deposited += max(0, self.previous_units[id]["cargo"]["ore"]-unit["cargo"]["ore"])
                ice_mined += max(0, unit["cargo"]["ice"]-self.previous_units[id]["cargo"]["ice"])
                ore_mined += max(0, unit["cargo"]["ore"]-self.previous_units[id]["cargo"]["ore"])

        self.previous_units = units
        return ice_deposited, ore_deposited, ice_mined, ore_mined
    

    def rubble_reward(self):
        obs = self.env.state.get_obs()
        rubble = obs["board"]["rubble"]
        rubble_mined = 0

        
        for id, action in self.prev_actions["player_0"].items():
            if isinstance(action, list):
                if action[0][0] == 3 and id in obs["units"]["player_0"].keys():
                    x, y = obs["units"]["player_0"][id]["pos"]
                    rubble_mined += max(self.previous_rubble[x, y] - rubble[x, y], 0) #Max of this and 0 because dying factories leave behind rubble, although this should be masked away
        self.previous_rubble = rubble
        
        return rubble_mined

    def reward(self, rewards):
        # NOTE: Only handles player_0 atm

        """if len(self.agents) == 0:  # Game is over
            strain_ids = self.state.teams["player_0"].factory_strains
            agent_lichen_mask = np.isin(
                self.state.board.lichen_strains, strain_ids
            )
            lichen = self.state.board.lichen[agent_lichen_mask].sum()
            # TODO: Check for correctness
            reward = self.config['scaling_win'] if rewards["player_0"] > rewards["player_1"] else -self.config['scaling_win'] if rewards["player_0"] < rewards["player_1"] else 0
            reward += np.tanh(lichen/self.config["lichen_divide_value"])*self.config['scaling_lichen']
            #self.env.state.stats["player_0"]['rewards']['end_of_episode_reward'] += reward
            return reward"""
        

        #Factories lost reward
        factories_lost = -(self.env.state.stats["player_0"]["destroyed"]["FACTORY"]-self.previous_destroyed_factories)*self.config["factory_lost"]
        self.previous_destroyed_factories = self.env.state.stats["player_0"]["destroyed"]["FACTORY"]


        #Resource reward
        ice_deposited, ore_deposited, ice_mined, ore_mined = self.transfer_reward()

        resource_reward = ice_mined*self.config["scaling_ice"] + ore_mined*self.config["scaling_ore"]+ice_deposited*self.config["scaling_water"] + ore_deposited*self.config["scaling_metal"]


        #Unit reward, only heavies atm
        rubble_reward = self.rubble_reward() if self.config["rubble_reward"] != 0 else 0

        #Normalize and return
        num_factories = self.env.state.board.factories_per_team

        resource_reward /= num_factories
        rubble_reward = (rubble_reward*self.config["rubble_reward"])/num_factories
        
        reward = (
            resource_reward
            + rubble_reward
            + factories_lost
        )

        self.env.state.stats["player_0"]["total_episodic_reward"] += reward
        self.env.state.stats["player_0"]['rewards']['resource_reward'] += resource_reward
        self.env.state.stats["player_0"]['rewards']['rubble_reward'] += rubble_reward
        self.env.state.stats["player_0"]["rewards"]["factory_punishment"] += factories_lost

        return reward