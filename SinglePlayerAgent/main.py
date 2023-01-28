from luxai_s2.env import LuxAI_S2
from agent import Agent
from utils.visualization import animate
from utils.wrappers import ImageWithUnitsWrapper, SinglePlayerEnv

def play_episode(agent, env, make_gif = True):
    obs, original_obs = env.reset()
    step = 0
    imgs = []
    while env.state.real_env_steps < 0:
        o = original_obs[agent.player]
        a = agent.early_setup(step, o)
        step += 1
        obs, rewards, dones, infos = env.step(a)
        if make_gif:
            imgs += [env.render("rgb_array", width=640, height=640)]
        obs, original_obs = obs
    done = False
    while not done:
        actions = agent.act(step, obs[agent.player])
        step += 1
        obs, rewards, done, infos = env.step(actions)
        obs, original_obs = obs
        if make_gif:
            imgs += [env.render("rgb_array", width=640, height=640)]

    if make_gif:
        animate(imgs)

def run():
    env = LuxAI_S2(verbose = 2)
    env = ImageWithUnitsWrapper(env)
    env = SinglePlayerEnv(env)
    agent = Agent("player_0", env.state.env_cfg)
    
    play_episode(agent, env, True)