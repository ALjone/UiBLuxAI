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
        self.units = {}
        self.factories = {}

    def reset(self):
        self.units = {}
        return super().reset()
    
    def reward(self, reward):
        # does not work yet. need to get the agent's observation of the current environment 
        obs_ = self.state.get_obs()
        obs = {}
        for k in self.agents:
            obs[k] = obs_
        if len(obs)<1:
            return -100
        agent = "player_0"
        shared_obs = obs["player_0"]

        # compute reward
        # we simply want to encourage the heavy units to move to ice tiles
        # and mine them and then bring them back to the factory and dump it
        # as well as survive as long as possible

        factories = shared_obs["factories"][agent]
        factory_pos = None
        for unit_id in factories:
            factory = factories[unit_id]
            # note that ice converts to water at a 4:1 ratio
            factory_pos = np.array(factory["pos"])
            break
        units = shared_obs["units"][agent]
        unit_deliver_ice_reward = 0
        unit_move_to_ice_reward = 0
        unit_overmining_penalty = 0
        delta_ice = 0
        power_reward = 0

        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)
        

        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        unit_power = 0
        for unit_id in units:
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            ice_tile_distances = np.mean((ice_tile_locations - pos) ** 2, 1)
            closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
            dist_to_ice = manhattan_dist(closest_ice_tile, pos)
            if factory_pos is not None:
                dist_to_factory = manhattan_dist(pos, factory_pos)
            else:
                dist_to_factory = np.inf
            unit_power = unit["power"]
            if unit_id in self.units.keys():
                prev_state = self.units[unit_id]
            else:
                prev_state = unit
            
            if unit_power <1:
                power_reward -=1
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
            + power_reward
        )
        reward = reward


        return reward
