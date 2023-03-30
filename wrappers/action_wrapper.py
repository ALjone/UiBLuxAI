import gym
import numpy.typing as npt
from gym import spaces
from actions.actions import outputs_to_actions

class action_wrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Tuple((spaces.Box(0, 31, shape = (48, 48)), spaces.Box(0, 4, shape = (48, 48))))#spaces.Dict({"player_0" : spaces.Tuple((spaces.Box(0, 31, shape = (48, 48)), spaces.Box(0, 4, shape = (48, 48))))})

    def transform_action(self, action, player):
        unit_output = action[0]
        factory_output = action[1]
        obs = self.last_obs["player_0"]
        units = obs["units"][player].values()
        factories = obs["factories"][player].values()

        state = self.last_state_p0 if player == "player_0" else self.last_state_p1

        action = outputs_to_actions(unit_output, factory_output, units, factories, state, obs) #Second channel is always factory_map

        return action  
    
    def action(self, action: npt.NDArray):
        #return self.transform_action(action, "player_0")
        #TODO: Change this when going to 2p

        # here, for each agent in the game we translate their action into a Lux S2 action
        lux_action = dict()
        for player in self.env.agents:
            if player == "player_0":
                lux_action[player] = self.transform_action(action, player)
            else:
                lux_action[player] = dict()

        """#TODO: Fix 
        agent = self.agents[0]
        units = self.env.state.units[agent]
        for unit_id, act in lux_action[agent].items():
            if "unit" in unit_id:
                act = act[0] #Because of action queue
                print(act)
                self.env.state.stats[agent]["actions"]["units"][act] += 1
                #Recharge action if the first element in the action array is 5
                if act == 0 and unit_id in units.keys(): #No idea why this check is needed??? Maybe it died
                    self.env.state.stats[agent]["actions"]["average_power_when_recharge"].append(units[unit_id].power)
            elif "factory" in unit_id:
                self.env.state.stats[agent]["actions"]["factories"][act] += 1"""
        
        # lux_action is now a dict mapping agent name to an action
        return lux_action