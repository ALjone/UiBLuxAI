import gym
from gym import spaces
from actions.actions import unit_output_to_actions, factory_output_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from actions.action_utils import where_will_unit_end_up

class action_wrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        #self.action_space = spaces.Tuple((spaces.Box(0, 31, shape = (self.config["map_size"], self.config["map_size"])), spaces.Box(0, 4, shape = (self.config["map_size"], self.config["map_size"]))))
        
        self.action_space = spaces.Dict({
                                            "player_0" : spaces.Dict({ 
                                                            "unit_action": spaces.Box(0, UNIT_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])), 
                                                            "factory_action": spaces.Box(0, FACTORY_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"]))
                                                        }),

                                            "player_1" : spaces.Dict({ 
                                                            "unit_action": spaces.Box(0, UNIT_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"])), 
                                                            "factory_action": spaces.Box(0, FACTORY_ACTION_IDXS, shape = (self.config["map_size"], self.config["map_size"]))
                                                        })
                                        })
        

    def transform_action(self, actions, player, obs):
        units = obs["units"][player].values()
        factories = obs["factories"][player].values()

        state = self.last_state_p0 if player == "player_0" else self.last_state_p1

        transformed_action = unit_output_to_actions(actions["unit_action"], units, state, obs["board"]["rubble"], obs["board"]["ice"], obs["board"]["ore"]) #Second channel is always factory_map

        transformed_action = transformed_action | factory_output_to_actions(actions["factory_action"], factories)

        return transformed_action
    
    def action(self, action):
        # here, for each agent in the game we translate their action into a Lux S2 action
        lux_action = dict()

        for player in self.env.agents:
            lux_action[player] = self.transform_action(action[player], player, self.last_obs[player])
        return lux_action