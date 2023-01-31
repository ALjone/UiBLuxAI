from train import train
from ppo import PPO
from luxai_s2.env import LuxAI_S2
from agents.RL_agent import Agent
from utils.visualization import animate
from wrappers.observation_wrappers import ImageWithUnitsWrapper
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from utils.utils import load_config
from gym.wrappers import RecordVideo

config = load_config()

env = LuxAI_S2(verbose=0, collect_stats=True)
env = ImageWithUnitsWrapper(env)
env = SinglePlayerEnv(env)
env = IceRewardWrapper(env, config)
env = RecordVideo(env, "videos", episode_trigger= lambda x : x % 999 == 0)
env.reset()
agent = Agent("player_0", env.state.env_cfg, config)

train(env, agent, config)
