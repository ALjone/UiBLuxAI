from luxai_s2.env import LuxAI_S2
from agents.RL_agent import Agent
from utils.visualization import animate
from wrappers.observation_wrapper import StateSpaceVol2
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from wrappers.action_wrapper import action_wrapper
from utils.utils import load_config
import torch
from utils.visualization import visualize_obs
import matplotlib.pyplot as plt

def play_episode(agent: Agent, env: LuxAI_S2, make_gif = True):
    obs = env.reset()
    step = 0
    map_imgs = []
    obs_imgs = []
    done = False
    while not done:
        i, g, um = obs
        actions = agent.get_action(i, g, um)
        step += 1

        if step % 50 == 0 and not make_gif:
            img = env.render("rgb_array", width=640, height=640)
            plt.imshow(img)
            plt.savefig(f'Images/map-{step}.png')
            plt.cla()
            plt.close()
            img = visualize_obs(i, step, g, True)
            obs_imgs.append(img)
        elif make_gif:
            img = visualize_obs(i, step, g)
            obs_imgs.append(img)

        obs, _, done, _ = env.step(actions)

        if make_gif:
            map_imgs += [env.render("rgb_array", width=640, height=640)]

    if make_gif:
        print("Making gif")
        animate(map_imgs, "map.gif")
        animate(obs_imgs, "observation.gif")

def run(config):
    #Always run visualization on CPU
    config["device"] = torch.device("cpu")
    env = LuxAI_S2(verbose=1, collect_stats=True, MIN_FACTORIES = config["n_factories"], MAX_FACTORIES = config["n_factories"], validate_action_space = True)
    env.reset() #NOTE: Reset here to initialize stats
    env = SinglePlayerEnv(env, config)
    env = StateSpaceVol2(env, config)
    env = IceRewardWrapper(env, config)
    env.reset()
    env = action_wrapper(env)
    env.reset()
    agent = Agent("player_0", config)
    
    play_episode(agent, env, True)


if __name__ == "__main__":
    config = load_config()
    config["mode_or_sample"] = "sample"
    run(config)