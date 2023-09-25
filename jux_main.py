import torch
a = torch.ones((5000, 5000, 190), device=torch.device("cuda"))
print("Size of a:", (a.nelement()*a.element_size())/10**9, "GB")
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".15"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["WANDB_SILENT"] = "true"
import jax

del a 

from jux_train import train_jux

from jux.env import JuxEnvBatch
from jux.config import JuxBufferConfig, EnvConfig
from agents.jux_agent import Agent
import numpy as np
from utils.utils import load_config



config = load_config()

env_cfg = EnvConfig(verbose=1, map_size = config["map_size"])
buf_cfg = JuxBufferConfig(MAX_N_UNITS=200)

jux_env_batch = JuxEnvBatch(
    env_cfg=env_cfg,
    buf_cfg=buf_cfg,
)

seeds = np.random.randint(0, 2**32-1, dtype=np.int64, size = config["parallel_envs"])

jux_env_batch.reset(seeds)
agent = Agent(jux_env_batch.env_cfg, config)
#TODO: Make another class to have a "frozen enemy"
#agents = [Agent(jux_env_batch.env_cfg, config), Agent(jux_env_batch.env_cfg, config)]
agents = [agent, agent]

train_jux(jux_env_batch, agents, config, seeds)
