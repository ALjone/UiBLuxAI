from luxai_s2.env import LuxAI_S2
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from animate import animate
from utils.wrappers import ImageAndPosToUnitIDWrapper

env = LuxAI_S2(verbose = 1)
env = ImageAndPosToUnitIDWrapper(env)

obs, original_obs = env.reset()


agents = {player: Agent(player, env.state.env_cfg) for player in env.agents}

step = 0
steps = 200
imgs = []
while env.state.real_env_steps < 0:
    actions = {}
    for player in env.agents:
        o = original_obs[player]
        a = agents[player].early_setup(step, o)
        actions[player] = a
    step += 1
    obs, rewards, dones, infos = env.step(actions)
    imgs += [env.render("rgb_array", width=640, height=640)]
    obs, original_obs = obs
done = False
while not done:
    if step >= steps: break
    actions = {}
    for player in env.agents:
        o = obs[player]
        a = agents[player].act(step, o)
        actions[player] = a
    step += 1
    obs, rewards, dones, infos = env.step(actions)
    obs, original_obs = obs
    done = dones["player_0"] and dones["player_1"]
    imgs += [env.render("rgb_array", width=640, height=640)]


animate(imgs)
