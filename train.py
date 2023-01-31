import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm


def skip_early_phase(env, agent):
        state, original_obs = env.reset()
        step = 0
        while env.state.real_env_steps < 0:
            o = original_obs[agent.player]
            a = agent.early_setup(step, o)
            step += 1
            state, rewards, dones, infos = env.step(a)
            state, original_obs = state

        return state

def train(env, agent, config, writer = None):
    
    #Set all used variables
    start_time = time.time()
    time_step = 0
    i_episode = 0
    print_running_reward = 0
    print_running_episodes = 0
    highest_reward = 0
    train_time = time.time()

    if writer is None:
        writer = SummaryWriter()
    # training loop
    for _ in range(config["max_episodes"]//config["print_freq"]):  

        losses = []
        for _ in tqdm(range(config["print_freq"]), leave = False, desc = "Experiencing"):
            current_ep_reward = 0
            state = skip_early_phase(env, agent)
            while True:
                # select action with policy
                action = agent.act(state)
                #a = ppo_agent.act(state)
                state, reward, done, _ = env.step(action)
                state = state[0] #Getting the first element, because state is a tuple of (state, original_obs)

                # saving reward and is_terminals
                agent.PPO.buffer.rewards.append(reward)
                agent.PPO.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % config["batch_size"] == 0:
                    losses.append(agent.PPO.update(state["image_features"].to(config["device"])))


                # break; if the episode is over
                if done:
                    break
                

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1


        # print average reward till last episode
        print_avg_reward = print_running_reward / print_running_episodes
        print_avg_reward = round(print_avg_reward, 7)

        print("Episode : {}/{} \t\t Timestep : {} \t\t Average Reward : {} \t\t Average loss : {} \t\t Time used last {} eps: {} \t\t Time used total: {}".format(i_episode, config["max_episodes"], time_step, print_avg_reward, np.mean(losses), config["print_freq"], round(time.time()-train_time, 1), round(time.time()-start_time, 1)))

        writer.add_scalar("Average reward", print_avg_reward, i_episode)

        print_running_reward = 0
        print_running_episodes = 0
        train_time = time.time()

        if print_avg_reward > highest_reward:
            highest_reward == print_avg_reward
            agent.PPO.save(config["save_path"])
