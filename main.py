from train import train
from ppo import PPO
from luxai_s2.env import LuxAI_S2
from agents.RL_agent import Agent
from utils.visualization import animate
from wrappers.observation_wrappers import  ImageWithUnitsWrapper
from wrappers.other_wrappers import SinglePlayerEnv
from utils.utils import load_config

config = load_config()

env = LuxAI_S2(verbose = 0, collect_stats=True)
env = ImageWithUnitsWrapper(env)
env = SinglePlayerEnv(env)
agent = Agent("player_0", env.state.env_cfg, config)

train(env, agent, config)