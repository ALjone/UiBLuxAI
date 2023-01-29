from ppo import PPO
from luxai_s2.env import LuxAI_S2
from agent import Agent
from utils.visualization import animate
from utils.wrappers import ImageWithUnitsWrapper, SinglePlayerEnv
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm


import io
import sys
import traceback


class TestableIO(io.BytesIO):

    def __init__(self, old_stream, initial_bytes=None):
        super(TestableIO, self).__init__(initial_bytes)
        self.old_stream = old_stream

    def write(self, bytes):
        if 'unit' in bytes:
            traceback.print_stack(file=self.old_stream)
        self.old_stream.write(bytes)


sys.stdout = TestableIO(sys.stdout)
sys.stderr = TestableIO(sys.stderr)




def train(env, agent, config, writer = None):

    #TODO: Move to config
    start_time = time.time()
    max_episodes = 10000000
    print_freq = 100        # print avg reward in the interval (in episodes)
    checkpoint_path = "model.t"
    batch_size = 256
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0
        
    train_time = time.time()
    if writer is None:
        writer = SummaryWriter()
    # training loop
    for i in range(max_episodes//print_freq):  

        losses = []
        for _ in tqdm(range(print_freq), leave = False, desc = "Experiencing"):
            state, original_obs = env.reset()
            step = 0
            while env.state.real_env_steps < 0:
                o = original_obs[agent.player]
                a = agent.early_setup(step, o)
                step += 1
                state, rewards, dones, infos = env.step(a)
                state, original_obs = state
            current_ep_reward = 0
            
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
                if time_step % batch_size == 0:
                    losses.append(agent.PPO.update())


                # break; if the episode is over
                if done:
                    break
                

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1


        # print average reward till last episode
        print_avg_reward = print_running_reward / print_running_episodes
        print_avg_reward = round(print_avg_reward, 7)

        print("Episode : {}/{} \t\t Timestep : {} \t\t Average Reward : {} \t\t Average loss : {} \t\t Time used last 100 eps: {} \t\t Time used total: {}".format(i_episode, max_episodes, time_step, print_avg_reward, np.mean(losses), round(time.time()-train_time, 1), round(time.time()-start_time, 1)))

        writer.add_scalar("Average reward", print_avg_reward, i_episode)

        print_running_reward = 0
        print_running_episodes = 0
        train_time = time.time()

        agent.PPO.save(checkpoint_path)
