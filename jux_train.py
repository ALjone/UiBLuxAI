import time
import numpy as np
from utils.utils import formate_time
from tqdm import tqdm
from utils.wandb_logging import WAndB
from actions.old.idx_to_lux_move import MOVE_NAMES
from jux.torch import from_torch
from jux_wrappers.observation_wrapper import StateProcessor
from actions.tensor_to_jux_action import jux_action
import jax.numpy as jnp
from jux.env import JuxEnvBatch
import time
from agents.jux_agent import Agent
from copy import deepcopy


def do_early_phase(env: JuxEnvBatch, agents, config, seeds):
    num_envs = config["parallel_envs"]
    #TODO: Not random
    state = env.reset(seeds)
    #state = env.reset(np.random.randint(0, 2**32-1, dtype=np.int64, size = num_envs))
    step = 1

    bid, faction = jnp.zeros((num_envs, 2)), jnp.zeros((num_envs, 2))
    state, _ = env.step_bid(state, bid, faction)
    spawn = np.zeros((num_envs, 2, 2), dtype=jnp.int8)
    water = np.zeros((num_envs, 2), dtype=jnp.int16)
    metal = np.zeros((num_envs, 2), dtype=jnp.int16)

    #They should always have same real_env_steps
    while (state.real_env_steps < 0).any():
        valid_spawns_mask = state.board.valid_spawns_mask
        s, w, m = agents[state.next_player[0]].early_setup(step, state, valid_spawns_mask)

        # TODO: fix
        spawn[:, state.next_player, :] = s
        water[:, state.next_player] = w
        metal[:, state.next_player] = m
        step += 1
        state, (observations, rewards, dones, infos) = env.step_factory_placement(state, spawn, water, metal)

    return state

def get_factory_action_dist(action):
    faction = action.factory_action
    f_actions = [jnp.sum(faction[:, 0, :] == -1).item(), jnp.sum(faction[:, 0, :] == 0).item(), jnp.sum(faction[:, 0, :] == 1).item(), jnp.sum(faction[:, 0, :] == 2).item()]
    return f_actions


def train_jux(env, agents: list[Agent], config, seeds):
    state_processor = StateProcessor(config)

    if (config["log_to_wb"]):
        wb = WAndB(config=config, run_name='Testing run')

    #init_state = do_early_phase(env, agents, config)
    for _ in range(config["max_episodes"]//config["print_freq"]):
        with tqdm(total = 100, desc = "Games played", leave = False) as pbar_outer:
            while True:
                #state = deepcopy(init_state)
                state = do_early_phase(env, agents, config, seeds)
                obs = state_processor.process_state(state)
                s = np.zeros(config["parallel_envs"], dtype=np.int16)
                state_processor.reset()
                train_time = 0
                action_transform_time = 0
                state_process_time = 0
                step_time = 0
                agent_act_time = 0
                backprop_time = 0
                gather_time = 0
                total_time = time.time()
                #TODO: Uncomment below
                reward = 0
                time_step = 0
                max_units = 0
                loss = 0
                Q_target = 0
                Q_next = 0
                Q_unit = 0
                Q_factory = 0
                training_reward = 0
                f_action_dist = [0, 0, 0, 0]
                with tqdm(total=1000, desc = "Stepping", leave = False) as pbar_inner:
                    while True:
                        time_step += 1
                        
                        start_time = time.time()
                        p0_unit_actions, p0_factory_actions = agents[0].act(state, obs[0][0], obs[0][1], 0)
                        p1_unit_actions, p1_factory_actions = agents[1].act(state, obs[1][0], obs[1][1], 1)
                        agent_act_time += time.time()-start_time

                        start_time = time.time()
                        action = jux_action(from_torch(p0_unit_actions), from_torch(p0_factory_actions), from_torch(p1_unit_actions), from_torch(p1_factory_actions), state)
                        action_transform_time +=  time.time()-start_time
                        single_f_action_dist = get_factory_action_dist(action)
                        f_action_dist = [f_action_dist[i] + single_f_action_dist[i] for i in range(4)]
                        
                        start_time = time.time()
                        state, (_, _, dones, _) = env.step_late_game(state, action)
                        step_time +=  time.time()-start_time
                        
                        pbar_inner.update(1)
                        old_obs = obs

                        start_time = time.time()
                        obs = state_processor.process_state(state)
                        state_process_time +=  time.time()-start_time
                        reward += (jnp.mean(obs[0][2])/(state.n_factories[0, 0]))
                        #TODO: Add in masking of games that end prematurely
                        
                        #Train only one agent, because the other is frozen, which is also why we only get p0 data
                        p0_obs = obs[0]
                        p0_old_obs = old_obs[0]
                        p0_dones = dones[:, 0]

                        start_time = time.time()
                        single_loss, single_q_target, single_q_next, single_reward, single_unit, single_factory, bp_time, tt_time = agents[0].TD.train2(p0_obs[0],
                                                                                                 p0_obs[1], p0_unit_actions, p0_factory_actions, p0_obs[2], p0_old_obs[0], p0_old_obs[1], p0_dones, state, 0)
                        train_time +=  time.time()-start_time
                        backprop_time += bp_time
                        gather_time += tt_time
                        loss += single_loss
                        Q_target += single_q_target
                        Q_next += single_q_next
                        training_reward += single_reward
                        Q_unit += single_unit
                        Q_factory += single_factory

                        if state.n_units[:, 0].max() > max_units:
                            max_units = state.n_units[:, 0].max()
                        #print("Single timestep reward:", (jnp.mean(obs[0][2])/(state.n_factories[0, 0])).item())

                        s += np.array(~dones[:, 0] + ~dones[:, 1], dtype=np.int16)
                        if dones.all() or time_step > 30: 
                            break

                #print("\n\nTotal time:", round(time.time()-total_time, 2), "seconds")
                #seconds = state_process_time + train_time + action_transform_time + step_time + agent_act_time
                #print("Sum of the below:", round(seconds, 2), "seconds")
                #print("Time to process env:", round(state_process_time, 2), "seconds")
                #print("Total train time:", round(train_time, 2), "seconds")
                #print("Time to train without backprop and no_grad part:", round(train_time-backprop_time-gather_time, 2), "seconds")
                #print("Time to backprop:", round(backprop_time, 2), "seconds")
                #print("Time to do first part:", round(gather_time, 2), "seconds")
                #print("Time to transform action:", round(action_transform_time, 2), "seconds")
                #print("Time to step:", round(step_time, 2), "seconds")
                #print("Time to act:", round(agent_act_time, 2), "seconds")
                pbar_outer.update(s.sum())
                pbar_outer.set_description(f"Avg reward per game: {round(reward.item(), 3)}, Steps played")

                if config["log_to_wb"]:
                    log_dict = {}

                    log_dict["Main/Average reward"] = reward.item()
                    log_dict["Main/Average episode length in batch"] = s.mean()
                    log_dict["Main/Max units at any time"] = max_units
                    log_dict["Training/Loss"] = loss/time_step
                    log_dict["Training/Target"] = Q_target/time_step
                    log_dict["Training/Next"] = Q_next/time_step
                    log_dict["Training/Reward"] = training_reward/time_step
                    log_dict["Training/Unit"] = Q_unit/time_step
                    log_dict["Training/Factory"] = Q_factory/time_step
                    log_dict["Distribution/Factory"] = f_action_dist
                    wb.log(log_dict)
