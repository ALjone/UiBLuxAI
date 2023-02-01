import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.utils import formate_time
from tqdm import tqdm
from utils.stat_collector import StatCollector

def do_early_phase(env, agent):
        state = env.reset()
        step = 0
        while env.state.real_env_steps < 0:
            a = agent.early_setup(step, state)
            step += 1
            state, rewards, dones, infos = env.step(a)

        return state

def train(env, agent, config, writer = None):
    assert env.collect_stats

    stat_collector = StatCollector("player_0")
    #Set all used variables
    start_time = time.time()
    time_step = 0
    i_episode = 0
    print_running_reward = 0
    print_running_episodes = 0
    highest_reward = -np.inf
    last_x_ep_time = time.time()

    if writer is None:
        writer = SummaryWriter()
    # training loop
    for _ in range(config["max_episodes"]//config["print_freq"]):  

        losses = []
        step_counter = 0
        for _ in tqdm(range(config["print_freq"]), leave = False, desc = "Experiencing"):
            current_ep_reward = 0
            state = do_early_phase(env, agent)
            while True:
                # select action with policy
                action = agent.act(state)
                #a = ppo_agent.act(state)
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                agent.PPO.buffer.rewards.append(reward)
                agent.PPO.buffer.is_terminals.append(done)

                time_step +=1
                step_counter += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % config["batch_size"] == 0:
                    losses.append(agent.PPO.update())


                # break; if the episode is over
                if done:
                    break
                
            stat_collector.update(env.state.stats)
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1


        # print average reward till last episode
        print_avg_reward = print_running_reward / print_running_episodes
        print_avg_reward = round(print_avg_reward, 7)

        steps_per_second = step_counter/(time.time()-last_x_ep_time)
        print(f"Episode : {i_episode:>6} \tTimestep : {time_step:>8} \tAverage Reward : {round(print_avg_reward, 3):>7} \t Average episode length: {round(step_counter/config['print_freq'], 3):>5}",
               f"\tAverage loss : {round(np.mean(losses).item(), 3):>6} \tSteps per second last {config['print_freq']:>5} eps: {int(steps_per_second):>4} \tTime used total: {formate_time(int(time.time()-start_time))}")

        print_running_reward = 0
        print_running_episodes = 0

        last_x_ep_time = time.time()

        if print_avg_reward > highest_reward:
            highest_reward == print_avg_reward
            agent.PPO.save(config["save_path"])


        writer.add_scalar("Main/Average reward", print_avg_reward, i_episode)
        writer.add_scalar("Main/Average episode length", step_counter/config['print_freq'], i_episode)
        writer.add_scalar("Main/Average steps per second", steps_per_second, i_episode)
        writer.add_scalar("Main/Average loss", np.mean(losses).item(), i_episode)

        #Update writer with average for last x eps
        categories = stat_collector.get_last_x(config["print_freq"]) 
        for category_name, category in categories.items():
            for name, value in category.items():
                writer.add_scalar(f"{category_name}/{name}", np.mean(value), i_episode)