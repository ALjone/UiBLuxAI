import gym
import numpy.typing as npt
from gym import spaces
from actions.actions import unit_output_to_actions

class action_wrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Tuple((spaces.Box(0, 31, shape = (48, 48)), spaces.Box(0, 4, shape = (48, 48))))#spaces.Dict({"player_0" : spaces.Tuple((spaces.Box(0, 31, shape = (48, 48)), spaces.Box(0, 4, shape = (48, 48))))})

    #0: LIGHT, 1: HEAVY, 2: LICHEN, 3: NOTHING
    def get_single_factory_action(self, factory, lights, heavies):
        metal = factory["cargo"]["metal"]
        water = factory["cargo"]["water"]
        power = factory["power"]
        num_factories = self.env.state.board.factories_per_team

        target_num_heavies = 1

        target_num_lights = 5

        if metal > 100 and power > 500 and heavies/num_factories < target_num_heavies:
            return 1

        if metal > 10 and power > 50 and lights/num_factories < target_num_lights:# and heavies/num_factories >= target_num_heavies:
            return 0
        
        if metal > 100 and power > 500:
            return 1

        if water > 200 and self.env.state.real_env_steps > 200:
            return 2

        return 3


    def get_factory_actions(self, obs):
        actions = {}
        factories = obs["factories"]["player_0"]
        lights = 0
        heavies = 0
        for unit in obs["units"]["player_0"].values():
            if unit["unit_type"] == "LIGHT":
                lights += 1
            else:
                heavies += 1
        
        for id, fac in factories.items():
            action = self.get_single_factory_action(fac, lights, heavies)
            if action == 3: continue
            actions[id] = action
        return actions


    def transform_action(self, action, player, obs):
        units = obs["units"][player].values()

        state = self.last_state_p0 if player == "player_0" else self.last_state_p1

        action = unit_output_to_actions(action, units, state, obs["board"]["rubble"]) #Second channel is always factory_map

        return action  
    
    def action(self, action: npt.NDArray):
        #return self.transform_action(action, "player_0")
        #TODO: Change this when going to 2p

        # here, for each agent in the game we translate their action into a Lux S2 action
        lux_action = dict()
        obs = self.last_obs["player_0"]

        for player in self.env.agents:
            if player == "player_0":
                unit_action = self.transform_action(action, player, obs)
                factory_acttion = self.get_factory_actions(obs)
                unit_action.update(factory_acttion)
                lux_action[player] = unit_action
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