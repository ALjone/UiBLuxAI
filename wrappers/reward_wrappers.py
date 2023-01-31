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
    
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self):
        self.units = {}
        self.number_of_units = {"player_0" : 0, "player_1" : 1}
        self.number_of_factories
        return super().reset()

    def get_died_units(self, player = "player_0"):
        self_destructs = 0
        units = 0
        factories = 0
        if player in self.prev_actions.keys():
            for action_dict in self.prev_actions[player]:
                print(action_dict)
                for unit_id, action in action_dict:
                    if "unit" in unit_id: units += 1
                    else: factories += 1
                    if action[0] == 4: self_destructs += 1 #Idx 4 is self destruct
    
    def reward(self, reward):
        # does not work yet. need to get the agent's observation of the current environment 
        obs_ = self.state.get_obs()
        obs = {}
        for k in self.agents:
            obs[k] = obs_
        if len(self.agents) == 0: #Game is over
            return 0
        agent = "player_0"
        shared_obs = obs["player_0"]
        factories = shared_obs["factories"][agent]
        units = shared_obs["units"][agent]
        ice_map = shared_obs["board"]["ice"]

        actions = self.prev_actions
        
        ice_picked_up = 0

        ice_tile_locations = np.argwhere(ice_map == 1)

        return 0 

        for unit_id in units:
            unit = units[unit_id]
            pos = np.array(unit["pos"])

            if unit_id in self.units.keys():
                prev_state = self.units[unit_id]
            else:
                prev_state = unit

            # reward for digging and dropping of ice
            scaling = 10

            if dist_to_ice < 1e-3:
                delta_ice += scaling*(unit["cargo"]["ice"] - prev_state["cargo"]["ice"]) # postive reward for digging ice.
            elif dist_to_factory <= 3:
                delta_ice -= scaling*(unit["cargo"]["ice"] - prev_state["cargo"]["ice"]) # postive reward for dopping ice at factory.
            else:
                delta_ice -= min(0,scaling*(unit["cargo"]["ice"] - prev_state["cargo"]["ice"])) # negative reward for dropping ice. 
            
            if unit["cargo"]["ice"] < 20:
                dist_penalty = dist_to_ice / (10) 
                unit_move_to_ice_reward += dist_penalty
            else:
                if factory_pos is not None:
                    dist_penalty =  dist_to_factory / 10
                    unit_deliver_ice_reward += dist_penalty # encourage unit to move back to factory
        
        #update prev state to current
        self.units = {unit_id :units[unit_id] for unit_id in  units}
        self.factories = {unit_id :factories[unit_id] for unit_id in  factories}

        reward = (
            0
            + unit_move_to_ice_reward
            + unit_deliver_ice_reward
            + unit_overmining_penalty
            + delta_ice
        )
        reward = reward


        return reward
