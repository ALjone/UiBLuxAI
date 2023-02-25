import torch
import time
import numpy as np
from utils.utils import formate_time
from tqdm import tqdm
from utils.stat_collector import StatCollector
from utils.wandb_logging import WAndB
from wandb.plot import bar
from wandb import Table as wbtable
from actions.idx_to_lux_move import MOVE_NAMES
from jux.torch import from_torch, to_torch
from jux_wrappers import observation_wrapper
from actions.tensor_to_jux_action import jux_action
import jax.numpy as jnp
from jux.env import JuxEnvBatch
import timeit
import time
from agents.RL_agent import Agent


def do_early_phase(env: JuxEnvBatch, agents, config):
    num_envs = config["parallel_envs"]
    state = env.reset(np.random.randint(0, 2**32-1, dtype=np.int64, size = num_envs))
    step = 1

    bid, faction = np.zeros((num_envs, 2)), jnp.zeros((num_envs, 2))
    state, _ = env.step_bid(state, bid, faction)
    spawn = np.zeros((num_envs, 2, 2), dtype=jnp.int8)
    water = np.zeros((num_envs, 2), dtype=jnp.int16)
    metal = np.zeros((num_envs, 2), dtype=jnp.int16)

    #They should always have same real_env_steps
    while (state.real_env_steps < 0).any():
        valid_spawns_mask = state.board.valid_spawns_mask
        #torch_state  = state._replace(env_cfg=None).to_torch()
        s, w, m = agents[state.next_player[0]].early_setup(step, state, valid_spawns_mask)

        # TODO: fix
        spawn[:, state.next_player, :] = s
        water[:, state.next_player] = w
        metal[:, state.next_player] = m
        step += 1
        state, (observations, rewards, dones, infos) = env.step_factory_placement(state, spawn, water, metal)

    return state


def train_jux(env, agents: list[Agent], config):
    for _ in range(config["max_episodes"]//config["print_freq"]):
        with tqdm(total = 100, desc = "Games played", leave = False) as pbar_outer:
            while True:
                state = do_early_phase(env, agents, config)
                obs = observation_wrapper.observation(state)
                s = 0
                with tqdm(total=1000, desc = "Stepping", leave = False) as pbar_inner:
                    while True:
                        s += 1
                        actions = []
                        for i, (agent, (image_features, global_features)) in enumerate(zip(agents, obs)):
                            actions += agent.act(state, image_features, global_features, i)

                        action = jux_action(*[from_torch(action) for action in actions], state)

                        state, (_, rewards, dones, _) = env.step_late_game(state, action)
                        pbar_inner.update(1)
                        old_obs = obs
                        obs = observation_wrapper.observation(state)
                        #TODO: Add in masking of games that end prematurely
                        for i, (new_obs, old_obs) in enumerate(zip(old_obs, obs)):
                            agents[0].TD.train(new_obs[0], new_obs[1], actions[i*2+0], actions[i*2+1], rewards[:, i], old_obs[0], old_obs[1], dones[:, i], state, i)
                        del actions
                        del new_obs
                        del old_obs
                        del rewards
                        if dones.all():
                            print(f"Played {s} steps")
                            break
                        del dones
                pbar_outer.update(config["parallel_envs"])

def train(env, agent, config):

    # Set all used variables
    start_time = time.time()
    time_step = 0
    i_episode = 0
    print_running_reward = 0
    print_running_episodes = 0
    highest_reward = -np.inf
    last_x_ep_time = time.time()

    if (config["log_to_wb"]):
        wb = WAndB(config=config, run_name='Testing run')
    # training loop
    for _ in range(config["max_episodes"]//config["print_freq"]):

        losses = []
        step_counter = 0
        for _ in tqdm(range(config["print_freq"]), leave=False, desc="Experiencing"):
            current_ep_reward = 0
            state = do_early_phase(env, agent)
            ep_timesteps = 0
            agent.reset()
            while True:
                # select action with policy
                action = agent.act(state)
                #a = ppo_agent.act(state)
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                agent.PPO.buffer.rewards.append(reward)
                agent.PPO.buffer.is_terminals.append(done)

                time_step += 1
                step_counter += 1
                ep_timesteps += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % config["batch_size"] == 0:
                    loss = agent.PPO.update()
                    losses.append(loss)

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

        # print average reward till last episode
        print_avg_reward = print_running_reward / print_running_episodes
        print_avg_reward = round(print_avg_reward, 7)

        steps_per_second = step_counter/(time.time()-last_x_ep_time)
        print(f"Episode : {i_episode:>6} \tTimestep : {time_step:>8} \tAverage Reward : {round(print_avg_reward, 3):>7} \t Average episode length: {round(step_counter/config['print_freq'], 3):>7}",
              f"\tAverage loss : {round(np.mean(losses).item(), 3):>6} \tSteps per second last {config['print_freq']:>5} eps: {int(steps_per_second):>4} \tTime used total: {formate_time(int(time.time()-start_time))}")

        print_running_reward = 0
        print_running_episodes = 0

        last_x_ep_time = time.time()

        if print_avg_reward > highest_reward:
            highest_reward == print_avg_reward
            agent.PPO.save(config["save_path"])


        if config["log_to_wb"]:
            log_dict = {}

            log_dict["Main/Average reward"] = print_avg_reward
            log_dict["Main/Average episode length"] = step_counter/config["print_freq"]
            log_dict["Main/Average steps per second"] = steps_per_second
            log_dict["Main/Average loss"] = np.mean(losses).item()

            # Update wandb with average for last x eps
            wb.log(log_dict)