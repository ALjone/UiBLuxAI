import torch
a = torch.ones((5, 5), device=torch.device("cuda"))
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".10"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


from train import train_jux

from jux.env import JuxEnvBatch
from jux.config import JuxBufferConfig, EnvConfig
from agents.RL_agent import Agent
import numpy as np
from utils.utils import load_config
import jax




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
agents = [agent, agent]

train_jux(jux_env_batch, agents, config)
