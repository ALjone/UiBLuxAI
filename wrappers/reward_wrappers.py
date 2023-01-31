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
        return super().reset()

    def get_died_units_and_factories(self, player="player_0"):
        # TODO: Differentiate between light and heavy robots
        self_destructs = 0
        units = 0
        factories = 0
        if player in self.prev_actions.keys():
            for action_dict in self.prev_actions[player]:
                for unit_id, action in action_dict:
                    if "unit" in unit_id:
                        units += 1
                    else:
                        factories += 1
                    if action[0] == 4:
                        self_destructs += 1  # Idx 4 is self destruct

        units_died = units - self.number_of_units[player]
        factories_died = factories - self.number_of_factories[player]
        self.number_of_units[player] = units
        self.number_of_factories[player] = factories
        return units_died, factories_died

    def reward(self, rewards):
        # does not work yet. need to get the agent's observation of the current environment
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
            reward = -1 if rewards < 0 else 1
            reward += np.tanh(lichen/self.config["lichen_divide_value"])*20
            return reward

        agent = "player_0"
        shared_obs = obs["player_0"]
        factories = shared_obs["factories"][agent]
        units = shared_obs["units"][agent]
        factory_pos = [factory["pos"] for _, factory in factories.items()]

        units_lost, factories_lost = self.get_died_units_and_factories()
        units_killed, _ = self.get_died_units_and_factories("player_1")

        unit_lost_reward = units_lost*-self.config["unit_lost_scale"]
        factories_lost_reward = factories_lost*- \
            self.config["factory_lost_scale"]

        units_killed_reward = units_killed*self.config["units_killed_scale"]
        resource_reward = 0
        for unit_id in units:
            unit = units[unit_id]
            pos = list(unit["pos"])

            if unit_id in self.units.keys():
                prev_state = self.units[unit_id]
            else:
                prev_state = unit

            # reward for digging and dropping of resources

            # Scaling for ice, ore
            scaling = [self.config["scaling_ice"], self.config["scaling_ore"]]
            scaling_delivery_extra = self.config["scaling_delivery_extra"]
            delta_res = 0
            for res, scale in zip(["ice", "ore"], scaling):
                # Dropping ice at factory
                if pos in factory_pos:
                    delta_res += scaling_delivery_extra*scale * \
                        (unit["cargo"][res] - prev_state["cargo"][res])

                # Picking up ice, or dropping it somewhere bad
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
            + factories_lost_reward
            + units_killed_reward
            + resource_reward
        )
        if reward != 0:
            print(reward)
        return reward
