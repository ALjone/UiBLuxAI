from train import train
from ppo import PPO
from luxai_s2.env import LuxAI_S2
from agent import Agent
from utils.visualization import animate
from utils.wrappers import ImageWithUnitsWrapper, SinglePlayerEnv


env = LuxAI_S2(verbose = 0, collect_stats=True)
env = ImageWithUnitsWrapper(env)
env = SinglePlayerEnv(env)
agent = Agent("player_0", env.state.env_cfg)#, path = checkpoint_path)

train(env, agent, {})