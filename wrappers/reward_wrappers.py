import gym

class SimpleRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        obs = self.state.get_obs()
        observations = {}
        for k in self.agents:
            observations[k] = obs
        return reward