from train import train
from ppo import PPO
from luxai_s2.env import LuxAI_S2
from agents.RL_agent import Agent
from wrappers.observation_wrappers import StateSpaceVol1
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from utils.utils import load_config



config = load_config()

env = LuxAI_S2(verbose=0, collect_stats=True, map_size = config["map_size"])
env = StateSpaceVol1(env, config)
env = SinglePlayerEnv(env)
env = IceRewardWrapper(env, config)
#env = RecordVideo(env, "videos", episode_trigger= lambda x : x % config["video_save_rate"] == 0 and x != 0)
env.reset()
agent = Agent("player_0", env.state.env_cfg, config)

train(env, agent, config)

#0. Time things
#1. CleanRL
#2. Single-player env -> Two-player env
#3. Parallelize over envs