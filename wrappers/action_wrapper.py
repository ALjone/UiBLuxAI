import gym
from gym import spaces
from actions.actions import unit_output_to_actions, factory_output_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS

class action_wrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
        self.action_space = spaces.Dict({
                                            "player_0" : spaces.Dict({ 
                                                            "factory_action": spaces.Box(0, FACTORY_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])),
                                                            "light_unit_action": spaces.Box(0, UNIT_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])),
                                                            "heavy_unit_action": spaces.Box(0, UNIT_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])) 
                                                        }),

                                            "player_1" : spaces.Dict({ 
                                                            "factory_action": spaces.Box(0, FACTORY_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])),
                                                            "light_unit_action": spaces.Box(0, UNIT_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])),
                                                            "heavy_unit_action": spaces.Box(0, UNIT_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])) 
                                                        })
                                        })
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.env.state.stats[self.agents[0]]["actions_cancelled_per_unit"] = 0

        return obs

    def transform_action(self, actions, player, obs):
        units = obs["units"][player].values()
        factories = obs["factories"][player].values()

        transformed_action = unit_output_to_actions(actions["light_unit_action"], actions["heavy_unit_action"], units) | factory_output_to_actions(actions["factory_action"], factories)

        return transformed_action
    
    def action(self, action):
        # here, for each agent in the game we translate their action into a Lux S2 action
        lux_action = dict()

        for player in self.env.agents:
            lux_action[player] = self.transform_action(action[player], player, self.last_obs[player])
        return lux_action