from ppo import PPO
from luxai_s2.env import LuxAI_S2
from agent import Agent
from utils.visualization import animate
from utils.wrappers import ImageWithUnitsWrapper, SinglePlayerEnv
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

start_time = time.time()

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


max_training_timesteps = 10000000


print_freq = 100        # print avg reward in the interval (in episodes)
log_freq = 1000 * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

checkpoint_path = "model.t"

update_timestep = 1000 

print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

env = LuxAI_S2(verbose = -1, collect_stats=True)
env = ImageWithUnitsWrapper(env)
env = SinglePlayerEnv(env)
ppo_agent = Agent("player_0", env.state.env_cfg)

writer = SummaryWriter()

train_time = time.time()

# training loop
while time_step <= max_training_timesteps:

    state, original_obs = env.reset()
    step = 0
    while env.state.real_env_steps < 0:
        o = original_obs[ppo_agent.player]
        a = ppo_agent.early_setup(step, o)
        step += 1
        state, rewards, dones, infos = env.step(a)
        state, original_obs = state
    current_ep_reward = 0
    
    while True:
        # select action with policy
        action = ppo_agent.act(state)
        #a = ppo_agent.act(state)
        state, reward, done, _ = env.step(action)
        state = state[0] #Getting the first element, because state is a tuple of (state, original_obs)

        # saving reward and is_terminals
        ppo_agent.PPO.buffer.rewards.append(reward)
        ppo_agent.PPO.buffer.is_terminals.append(done)

        time_step +=1
        current_ep_reward += reward

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.PPO.update()


        # break; if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1

    # printing average reward
    if i_episode % print_freq == 0:

        # print average reward till last episode
        print_avg_reward = print_running_reward / print_running_episodes
        print_avg_reward = round(print_avg_reward, 7)

        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Time used last 100 eps: {} \t\t Time used total: {}".format(i_episode, time_step, print_avg_reward, round(time.time()-start_time, 1), round(time.time()-train_time, 1)))

        writer.add_scalar("Average reward", print_avg_reward, i_episode)

        print_running_reward = 0
        print_running_episodes = 0
        train_time = time.time()
