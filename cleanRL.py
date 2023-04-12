
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from utils.stat_collector import StatCollector
from utils.utils import load_config
import time
import gym
from tqdm import tqdm

from actions.actions import UNIT_ACTION_IDXS

from agents.RL_agent import Agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from luxai_s2.env import LuxAI_S2
from wrappers.observation_wrapper import StateSpaceVol2
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from wrappers.action_wrapper import action_wrapper
from utils.wandb_logging import WAndB
import os
os.environ["WANDB_SILENT"] = "true"
#torch.backends.cudnn.benchmark = True
#torch.backends.cuda.matmul.allow_tf32 = True


def make_env(config):
    def thunk():
        env = LuxAI_S2(verbose=0, collect_stats=True, MIN_FACTORIES = config["n_factories"], MAX_FACTORIES = config["n_factories"], validate_action_space = False)
        env.reset() #NOTE: Reset here to initialize stats
        env = SinglePlayerEnv(env, config)
        env = StateSpaceVol2(env, config)
        env = IceRewardWrapper(env, config)
        env.reset()
        env = action_wrapper(env)
        env.reset()
        return env

    return thunk


if __name__ == "__main__":
    config = load_config()
    run_name = str(time.time())
    if config["log_to_wb"]:
        logger = WAndB(config, run_name="CleanRL-run")

    stat_collector = StatCollector("player_0")

    device = config["device"]
    
    print("NOTE: Currently running without power cost.")    
    print("Running with mini batch size:", config["mini_batch_size"])


    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config) for _ in range(config["parallel_envs"])]
    )

    agent = Agent("player_0", config)
    optimizer = optim.Adam(agent.model.parameters(), lr=config["lr"], eps=1e-5)

    # ALGO Logic: Storage setup
    image_obs = torch.zeros((config["batch_size"], config["parallel_envs"]) + envs.observation_space[0].shape[1:]).to(device)
    global_obs = torch.zeros((config["batch_size"], config["parallel_envs"]) + envs.observation_space[1].shape[1:]).to(device)
    unit_action_masks = torch.zeros((config["batch_size"], config["parallel_envs"]) + (48, 48, UNIT_ACTION_IDXS), dtype=torch.bool).to(device)
    unit_actions = torch.zeros((config["batch_size"], config["parallel_envs"]) + (48, 48)).to(device)
    unit_logprobs = torch.zeros((config["batch_size"], config["parallel_envs"])).to(device)
    rewards = torch.zeros((config["batch_size"], config["parallel_envs"])).to(device)
    dones = torch.zeros((config["batch_size"], config["parallel_envs"])).to(device)
    values = torch.zeros((config["batch_size"], config["parallel_envs"])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    global_games_played = 0
    last_steps_played = 0
    last_games_played = 0
    higest_average = 0

    start_time = time.time()
    next_obs = envs.reset()
    next_image_obs, next_global_obs = torch.tensor(next_obs[0]).to(config["device"]), torch.tensor(next_obs[1]).to(config["device"])
    unit_action_mask = torch.tensor(next_obs[2], dtype=torch.bool).to(config["device"])
    next_done = torch.zeros(config["parallel_envs"]).to(device)
    num_updates = config["max_episodes"] // config["batch_size"]
    episode_lengths = []    
    episode_rewards = []


    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["lr"]
            optimizer.param_groups[0]["lr"] = lrnow

        stepping_time = 0
        pre_and_post_step = time.time()
        with tqdm(total = config["num_steps_per_env"]*config["parallel_envs"], desc = "Playing", leave = False) as pbar:
            for step in range(0, config["num_steps_per_env"]):
                global_step += 1 * config["parallel_envs"]
                image_obs[step] = next_image_obs
                global_obs[step] = next_global_obs
                dones[step] = next_done
                unit_action_masks[step] = unit_action_mask

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    unit_action, unit_logprob, _, value = agent.get_action_and_value(next_image_obs, next_global_obs, unit_action_mask)
                    values[step] = value.flatten()
                unit_actions[step] = unit_action
                unit_logprobs[step] = unit_logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                #TODO: This needs to be changed drastically when going to 2p
                action = unit_action.cpu().numpy()
                single_stepping_time = time.time()
                next_obs, reward, done, info = envs.step(action)
                stepping_time += time.time() - single_stepping_time
                #Collect stats:
                for d, single_info in zip(done, info):
                    if d:
                        stat_collector.update(single_info["stats"])
                        episode_lengths.append(single_info["episode_length"])
                        episode_rewards.append(single_info["stats"]["total_episodic_reward"])

                next_image_obs, next_global_obs = torch.tensor(next_obs[0]).to(config["device"]), torch.tensor(next_obs[1]).to(config["device"])
                unit_action_mask = torch.tensor(next_obs[2], dtype=torch.bool).to(config["device"])
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.Tensor(done).to(device)
                global_games_played += next_done.sum()
                last_steps_played += config["parallel_envs"]
                last_games_played += next_done.sum().item()

                pbar.update(config["parallel_envs"])
        pre_and_post_step = time.time()-pre_and_post_step
        training_time = time.time()
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_image_obs, next_global_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config["batch_size"])):
                if t == config["batch_size"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["lmbda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_image_obs = image_obs.reshape((-1,) + envs.observation_space[0].shape[1:])
        b_global_obs = global_obs.reshape((-1,) + envs.observation_space[1].shape[1:])
        b_unit_masks = unit_action_masks.reshape((-1,) + (48, 48, UNIT_ACTION_IDXS))
        b_unit_logprobs = unit_logprobs.reshape(-1)
        b_unit_actions = unit_actions.reshape((-1,) + (48, 48))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config["batch_size"])
        clipfracs = []
        for epoch in range(config["epochs_per_batch"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["mini_batch_size"]):
                end = start + config["mini_batch_size"]
                mb_inds = b_inds[start:end]

                _, unit_newlogprob, unit_entropy, newvalue = agent.get_action_and_value(b_image_obs[mb_inds],
                                                                                        b_global_obs[mb_inds],
                                                                                        b_unit_masks[mb_inds],
                                                                                        action_unit = b_unit_actions.long()[mb_inds])
                mb_advantages = b_advantages[mb_inds]
                if config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                #UNITS
                logratio = unit_newlogprob - b_unit_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["eps_clip"]).float().mean().item()]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["eps_clip"], 1 + config["eps_clip"])
                pg_unit_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config["clip_loss_value"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config["eps_clip"],
                        config["eps_clip"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = unit_entropy.mean()
                loss = pg_unit_loss - config["ent_coef"] * entropy_loss + v_loss * config["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), config["max_grad_norm"])
                optimizer.step()

            if config["use_target_kl"]:
                if approx_kl > config["target_kl_max"] or approx_kl < config["target_kl_min"]:
                    break
        
        training_time = time.time()-training_time
        pre_and_post_step -= stepping_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        


        if last_steps_played > config["log_rate"]:
            agent.save("most_recent.t")
            print(f"SPS: {int(last_steps_played / (time.time() - start_time))} Reward per timestep: {round(np.mean(episode_rewards).item(), 2)} Games played: {int(global_games_played)}")
            if np.mean(episode_rewards) > higest_average:
                higest_average = np.mean(episode_rewards)
                agent.save(config["save_path"])
            if config["log_to_wb"]:
                log_dict = {}
                log_dict["Main/Average reward per step"] = np.mean(episode_rewards).item()
                log_dict["Main/Average episode length"] = np.mean(episode_lengths)
                #log_dict["Main/Max episode length"] = np.max(episode_lengths)
                log_dict["Losses/value_loss"] = v_loss.item()
                log_dict["Losses/policy_loss"] = (pg_unit_loss).item()
                log_dict["Losses/entropy"] = entropy_loss.item()
                log_dict["Losses/old_approx_kl"] = old_approx_kl.item()
                log_dict["Losses/approx_kl"] = approx_kl.item()
                log_dict["Losses/clipfrac"] = np.mean(clipfracs)
                log_dict["Losses/explained_variance"] = explained_var
                log_dict["Charts/learning_rate"] = optimizer.param_groups[0]["lr"]
                log_dict["Charts/Steps per second"] = int(last_steps_played / (time.time() - start_time))
                log_dict["Charts/Portion of time as training"] = training_time/(training_time+pre_and_post_step+stepping_time)
                log_dict["Charts/Portion of time as misc"] = pre_and_post_step/(training_time+pre_and_post_step+stepping_time)
                log_dict["Charts/Portion of time in env.step "] = stepping_time/(training_time+pre_and_post_step+stepping_time)
                categories = stat_collector.get_last_x(int(last_games_played))
                for category_name, category in categories.items():
                    for name, value in category.items():
                        if name == "unit_action_distribution":
                            log_dict[f"{category_name}/{name}"] = value
                        elif name == "factory_action_distribution":
                            log_dict[f"{category_name}/{name}"] = value
                        else:
                            log_dict[f"{category_name}/{name}"] = np.mean(value)

                logger.log(log_dict, global_step)
            last_steps_played = 0
            last_games_played = 0
            start_time = time.time()
            episode_rewards = []
            episode_lengths = []
    print("Finished training")
    envs.close()
