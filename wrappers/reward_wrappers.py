import gym

class SimpleRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward