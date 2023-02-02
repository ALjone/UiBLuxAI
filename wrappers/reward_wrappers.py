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

    def reset(self):
        self.units = {}
        # TODO: Update these
        self.number_of_units = {"player_0": 0, "player_1": 0}
        self.number_of_factories = {"player_0": 0, "player_1": 0}
        return_val =  super().reset()
        self.env.state.stats["player_0"]['rewards'] = {'unit_lost_reward':0, 'factories_lost_reward':0, 'units_killed_reward':0, 'resource_reward': 0, 'end_of_episode_reward': 0}
        return return_val
    
    def get_died_units_and_factories(self, player="player_0"):
        # TODO: Differentiate between light and heavy robots
        # NOTE: Does not currently work for factories
        # TODO: Differantiate between killing and dying in the future
        #self_destructs = 0
        units = 0
        factories = 0
        units_made = 0
        if player in self.prev_actions.keys() and self.state.real_env_steps > 0:
            for unit_id, action in self.prev_actions[player].items():
                #print(unit_id)
                if "unit" in unit_id:
                    units += 1
                    #if action[0] == 4:
                    #    self_destructs += 1  # Idx 4 is self destruct
                elif "factory" in unit_id:
                    factories += 1
                    if action in [0, 1]: #NOTE: Only works if action masking
                        units_made += 1
                else:
                    raise ValueError("Oh no, what's wrong with this unit ID:", unit_id)

            units_died = (self.number_of_units[player] + units_made) - units # This is acctually just change of unit number
            #TODO fix this (This is based on actions, and factories that do nothing submit no actions)

            factories_died = 0#self.number_of_factories[player] - factories

            self.number_of_units[player] = units
            self.number_of_factories[player] = factories

            # In order to reward building units we allow units died to be negative for now
            #return max(0, units_died), max(0, factories_died)
            return units_died, max(0, factories_died)
        return 0, 0

    def reward(self, rewards):
        # does not work yet. need to get the agent's observation of the current environment
        # NOTE: Only handles player_0 atm
        obs_ = self.state.get_obs()
        obs = {}
        for k in self.agents:
            obs[k] = obs_
        if len(self.agents) == 0:  # Game is over
            strain_ids = self.state.teams["player_0"].factory_strains
            agent_lichen_mask = np.isin(
                self.state.board.lichen_strains, strain_ids
            )
            lichen = self.state.board.lichen[agent_lichen_mask].sum(
            )
            # TODO: Check for correctness
            reward = self.config['scaling_win'] if rewards["player_0"] > rewards["player_1"] else -self.config['scaling_win'] if rewards["player_0"] < rewards["player_1"] else 0
            reward += np.tanh(lichen/self.config["lichen_divide_value"])*self.config['scaling_lichen']
            self.env.state.stats["player_0"]['rewards']['end_of_episode_reward'] += reward
            return reward

        agent = "player_0"

        # Getting factory and unit numbers + factory positions
        shared_obs = obs["player_0"]
        factories = shared_obs["factories"][agent]
        units = shared_obs["units"][agent]
        factory_pos = [factory["pos"] for _, factory in factories.items()]

        # Getting units lost and units enemy has lost
        units_lost, factories_lost = self.get_died_units_and_factories()
        units_killed, _ = self.get_died_units_and_factories("player_1")

        unit_lost_reward = units_lost*-self.config["unit_lost_scale"] if units_lost < 0 else units_lost*-self.config["unit_lost_scale"]*self.config['birth_kill_relation']
        factories_lost_reward = factories_lost*- \
            self.config["factory_lost_scale"]
        # TODO: Implement this
        units_killed_reward = 0 #units_killed*self.config["units_killed_scale"]


        resource_reward = 0
        for unit_id, unit in units.items():
            pos = list(unit["pos"])

            if unit_id in self.units.keys():
                prev_state = self.units[unit_id]
            else:
                prev_state = unit

            # reward for digging and dropping of resources

            # Scaling for ice, ore
            scaling = [self.config["scaling_ice"], self.config["scaling_ore"]]
            delta_res = 0
            #TODO: Tripple check this
            for res, scale in zip(["ice", "ore"], scaling):
                # Dropping res at factory
                #NOTE: Prev - unit, because we want to currently have less than we had
                if pos in factory_pos:
                    delta_res += self.config["scaling_delivery_extra"]*scale * \
                        max((prev_state["cargo"][res] - unit["cargo"][res]), 0)
                # Picking up res, or dropping it somewhere bad
                else:
                    delta_res += scale * \
                        (unit["cargo"][res] - prev_state["cargo"][res])
            resource_reward += delta_res

        # update prev state to current
        self.units = {unit_id: units[unit_id] for unit_id in units}
        self.factories = {unit_id: factories[unit_id] for unit_id in factories}
        
        reward = (
            0
            + unit_lost_reward
            #+ factories_lost_reward
            #+ units_killed_reward
            + resource_reward
        )
        self.env.state.stats["player_0"]['rewards']['unit_lost_reward'] += unit_lost_reward
        self.env.state.stats["player_0"]['rewards']['factories_lost_reward'] += factories_lost_reward
        self.env.state.stats["player_0"]['rewards']['units_killed_reward'] += units_killed_reward
        self.env.state.stats["player_0"]['rewards']['resource_reward'] += resource_reward
        return reward
