from luxai_s2.env import LuxAI_S2
from agents.RL_agent import Agent
from utils.visualization import animate
from wrappers.observation_wrapper import StateSpaceVol2
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from wrappers.action_wrapper import action_wrapper
from utils.utils import load_config
import torch
from tqdm import tqdm
def play_episode(agent: Agent, env: LuxAI_S2, device):
    obs = env.reset()
    done = False
    reward = 0
    steps = 0
    while not done:
        i, g, um = obs
        steps += 1
        actions = agent.get_action(i.to(device), g.to(device), um.to(device))
        obs, r, done, _ = env.step(actions)
        reward += r
    return reward, steps
def run(config):
    #Always run visualization on CPU
    config["device"] = torch.device("cuda")
    env = LuxAI_S2(verbose=0, collect_stats=True, MIN_FACTORIES = config["n_factories"], MAX_FACTORIES = config["n_factories"], validate_action_space = True)
    env.reset() #NOTE: Reset here to initialize stats
    env = SinglePlayerEnv(env, config)
    env = StateSpaceVol2(env, config)
    env = IceRewardWrapper(env, config)
    env.reset()
    env = action_wrapper(env)
    env.reset()
    agent = Agent("player_0", config)
    
    total_reward = 0
    total_steps = 0
    for _ in tqdm(range(100), leave=False, desc="Playing games"):
        reward, steps = play_episode(agent, env, config["device"])
        total_reward += reward
        total_steps += steps
    print("Average reward over 100 episodes:", round(total_reward/100, 1))
    print("Average steps over 100 episodes:", round(total_steps/100, 1))
if __name__ == "__main__":
    config = load_config()
    config["mode_or_sample"] = "mode"
    run(config)