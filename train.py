import time
import numpy as np
from utils.utils import formate_time
from tqdm import tqdm
from utils.stat_collector import StatCollector
from utils.wandb_logging import WAndB
from wandb.plot import bar
from wandb import Table as wbtable
from actions.idx_to_lux_move import MOVE_NAMES


def do_early_phase(env, agents, config):
    num_envs = config["parallel_envs"]
    state = env.reset(np.random.randint(0, 2**32-1, dtype=np.int64, size = num_envs))
    
    #valid_factory_mask = np.array(state.board.valid_spawns_mask)
    #NOTE: The way below is 10-20% faster than the way above, and about 250% faster than using the built in state.board.valid_spawns all the way through
    valid_spawns_mask = np.array(~state.board.map.ice & ~state.board.map.ore) 
    valid_spawns_mask = valid_spawns_mask & np.roll(valid_spawns_mask, 1, axis=1)
    valid_spawns_mask = valid_spawns_mask & np.roll(valid_spawns_mask, -1, axis=1)
    valid_spawns_mask = valid_spawns_mask & np.roll(valid_spawns_mask, 1, axis=2)
    valid_spawns_mask = valid_spawns_mask & np.roll(valid_spawns_mask, -1, axis=2)
    valid_spawns_mask[:, [0, -1], :] = False
    valid_spawns_mask[:, :, [0, -1]] = False
    step = 1

    bid, faction = np.zeros((num_envs, 2)), np.zeros((num_envs, 2))
    state, _ = env.step_bid(state, bid, faction)
    spawn = np.zeros((num_envs, 2, 2), dtype=np.int8)
    water = np.zeros((num_envs, 2), dtype=np.int8)
    metal = np.zeros((num_envs, 2), dtype=np.int8)

    #They should always have same real_env_steps
    while (state.real_env_steps < 0).any():
        #valid_spawns_mask = state.board.valid_spawns_mask
        s, w, m = agents[state.next_player[0]].early_setup(step, state, valid_spawns_mask)
        
        spawn[:, state.next_player] = s
        water[:, state.next_player] = w
        metal[:, state.next_player] = m

        x = np.expand_dims(s[:, 0], axis = -1)
        y = np.expand_dims(s[:, 1], axis = -1)
        print("Valid mask shape:", valid_spawns_mask.shape)
        print("x shape", x.shape)
        print("Clip shape:", np.clip(x-6, a_min = 0, a_max = None).shape)

        valid_spawns_mask[:, np.array([1, 2]) : np.array([3, 4]), np.array([1, 2]) : np.array([3, 4])]
        #The valid spawns mask in Jux is slow, so we make our own fast one
        valid_spawns_mask[:, np.clip(x-6, a_min = 0, a_max = None, dtype=np.int8) : np.clip(x+6+1, a_min = None, a_max = config["map_size"]-1, dtype=np.int8),
                           np.clip(y-6, a_min = 0, a_max = None, dtype=np.int8) : np.clip(y+6+1, a_min = None, a_max = config["map_size"]-1, dtype=np.int8)] = False

        step += 1
        state, (observations, rewards, dones, infos) = env.step_factory_placement(state, spawn, water, metal)

    return state


def train_jux(env, agents, config):
    for _ in range(config["max_episodes"]//config["print_freq"]):
        for _ in tqdm(range(config["print_freq"]), leave=False, desc="Experiencing"):
            state = do_early_phase(env, agents, config)

            #TODO: The states can be stacked if the agent uses the same network
            #for agent in agents:
            #    action = agent.act(state)

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