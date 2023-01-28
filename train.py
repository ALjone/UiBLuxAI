from ppo import PPO
from luxai_s2.env import LuxAI_S2
from agent import Agent
from utils.visualization import animate
from utils.wrappers import ImageWithUnitsWrapper, SinglePlayerEnv
import time
from datetime import datetime

start_time = time.time()



max_training_timesteps = 100000


print_freq = 150 * 1        # print avg reward in the interval (in num timesteps)
log_freq = 1000 * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

checkpoint_path = "model.t"

update_timestep = 1000 

print_running_reward = 0
print_running_episodes = 1

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

env = LuxAI_S2(verbose = 0, collect_stats=True)
env = ImageWithUnitsWrapper(env)
env = SinglePlayerEnv(env)
ppo_agent = Agent("player_0", env.state.env_cfg)

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


        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 7)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        # break; if the episode is over
        if done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1