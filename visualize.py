from luxai_s2.env import LuxAI_S2
from agents.jux_agent import Agent
from utils.visualization import animate
from wrappers.observation_wrappers import ImageWithUnitsWrapper, StateSpaceVol1
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from utils.utils import load_config

def play_episode(agent, env, make_gif = True):
    agent.reset()
    obs = env.reset()
    step = 0
    imgs = []
    while env.state.real_env_steps < 0:
        a = agent.early_setup(step, obs)
        step += 1
        obs, rewards, dones, infos = env.step(a)
        if make_gif:
            imgs += [env.render("rgb_array", width=640, height=640)]
    done = False
    while not done:
        actions = agent.act(obs)
        step += 1
        obs, rewards, done, infos = env.step(actions)
        if make_gif:
            imgs += [env.render("rgb_array", width=640, height=640)]
    if make_gif:
        print("Making gif")
        animate(imgs)

def run(config):
    env = LuxAI_S2(verbose=0, collect_stats=True)
    env = StateSpaceVol1(env)
    env = SinglePlayerEnv(env)
    env = IceRewardWrapper(env, config)
    agent = Agent("player_0", env.state.env_cfg, config)
    
    play_episode(agent, env, True)


if __name__ == "__main__":
    config = load_config()
    run(config)