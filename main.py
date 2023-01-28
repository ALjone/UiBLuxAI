from luxai_s2.env import LuxAI_S2
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from animate import interact
from wrappers.obs_wrappers import ImageObservationWrapper

env = LuxAI_S2()
env.reset()

#env = ImageObservationWrapper(env)


# recreate our agents and run
agents = {player: Agent(player, env.state.env_cfg) for player in env.agents}
interact(env, agents, 1000, seed=2)
