from train import train_jux

from jux.env import JuxEnv, JuxEnvBatch
from jux.config import JuxBufferConfig, EnvConfig
from agents.RL_agent import Agent
import numpy as np
from utils.utils import load_config
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

config = load_config()

env_cfg = EnvConfig(verbose=1, map_size = config["map_size"])
buf_cfg = JuxBufferConfig(MAX_N_UNITS=200)

jux_env = JuxEnv(
    env_cfg=env_cfg,
    buf_cfg=buf_cfg,
)

jux_env_batch = JuxEnvBatch(
    env_cfg=env_cfg,
    buf_cfg=buf_cfg,
)

seeds = np.random.randint(0, 2**32-1, dtype=np.int64, size = config["parallel_envs"])

jux_env_batch.reset(seeds)

agents = [Agent(player, jux_env_batch.env_cfg, config) for player in [0, 1]]

train_jux(jux_env_batch, agents, config)
