import time
from luxai_s2.env import LuxAI_S2
import matplotlib.pyplot as plt
import numpy as np
from utils.visualization import animate
from wrappers.observation_wrappers import ImageWithUnitsWrapper
from utils.utils import formate_time
from tqdm import tqdm


def get_winner(rewards):
    if rewards["player_0"] > rewards["player_1"]:
        return 0
    if rewards["player_1"] > rewards["player_0"]:
        return 1 
    return 2


def run_match(env, agents):
    """Returns 0 for p1 win, 1 for p2 win, 2 for tie"""
    obs = env.reset()
    step = 0
    while env.state.real_env_steps < 0:
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
    done = False
    while not done:
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a
        step += 1
        obs, rewards, dones, infos = env.step(actions)
        done = dones["player_0"] or dones["player_1"]
    
    for reward in rewards.values():
        if reward != -1000 and reward != 0:
            print(rewards)

    return get_winner(rewards)


def run_tournament(env, agents, matches, names):
    start_time = time.time()
    win_rates = [0, 0, 0]
    for _ in tqdm(range(matches), leave=False, desc="Playing tournament"):
        winner = run_match(env, agents)
        win_rates[winner] += 1

    print(f"Played a tournament of {matches} matches in {formate_time(int(time.time()-start_time))}.\n\t{names[0]}",
          f"winrate: {round((win_rates[0]/matches)*100, 2)}%\n\t{names[1]} winrate: {round((win_rates[1]/matches)*100, 2)}%\n\tTies: {round((win_rates[2]/matches)*100, 2)}%")
