from luxai_s2.env import LuxAI_S2
from agents.heuristics.heuristic_agent_conv_v1 import Agent as conv_v1
from agents.heuristics.heuristic_agent_random_early import Agent as random_early
from agents.heuristics.heuristic_agent_conv_v2 import Agent as conv_v2
from tournaments.one_vs_one import run_tournament

env = LuxAI_S2(verbose = 0)

agents = {"player_0": random_early("player_0", env.state.env_cfg),
          "player_1": conv_v1("player_1", env.state.env_cfg)}


run_tournament(env, agents, 1000, ["Random", "Conv v1"])

print("\n\n")
agents = {"player_0": random_early("player_0", env.state.env_cfg),
          "player_1": conv_v2("player_1", env.state.env_cfg)}


run_tournament(env, agents, 1000, ["Random", "Conv v2"])

print("\n\n")
agents = {"player_0": conv_v1("player_0", env.state.env_cfg),
          "player_1": conv_v2("player_1", env.state.env_cfg)}


run_tournament(env, agents, 1000, ["Conv v1", "Conv v2"])