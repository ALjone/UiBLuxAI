
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from agents.opponent import Opponent
from utils.stat_collector import StatCollector
from utils.utils import load_config, save_with_retry, rotate_tensor
import time
import gym
from tqdm import tqdm
from collections import deque
import random
from pathlib import Path
from actions.actions import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS

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

def make_env(config, seed):
    def thunk():
        env = LuxAI_S2(verbose=0, collect_stats=True, map_size = config["map_size"], MIN_FACTORIES = config["n_factories_min"], MAX_FACTORIES = config["n_factories_max"], validate_action_space = False)
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
    print("Running with mini batch size:", config["mini_batch_size"], "Batch size:", config["batch_size"])

    if not (config["mini_batch_size"] & (config["mini_batch_size"]-1) == 0):
        print("NBNBNBNB!!! Running with a non-power-of-two batch size!")
    print("Will run:", config["total_timesteps"] // config["batch_size"], "updates.")


    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config, i) for i in range(config["parallel_envs"])]
    )

    agent = Agent(config)

    Path("Opponents/").mkdir(exist_ok=True)
    #agent.save("Opponents/start.t")
    opponent = Opponent(config)
    opponent.load(agent.model.state_dict())
    #possible_opponents = deque(maxlen=config["num_opponents"])
    #possible_opponents.append("Opponents/start.t")


    optimizer = optim.Adam(agent.model.parameters(), lr=config["lr"], eps=1e-5)

    image_obs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + envs.observation_space["player_0"]["features"].shape[1:]).to(device)

    #Unit
    unit_action_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"], UNIT_ACTION_IDXS)).to(device)
    unit_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)

    #Light unit
    light_unit_actions = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    light_unit_logprobs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)

    #Heavy unit
    heavy_unit_actions = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    heavy_unit_logprobs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)

    #Factory
    factory_action_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"], FACTORY_ACTION_IDXS)).to(device)
    factory_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    factory_actions = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    factory_logprobs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)

    rewards = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    dones = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    values = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    global_games_played = 0
    last_steps_played = 0
    last_games_played = 0
    higest_average = 0
    steps_since_opponent_update = 0

    start_time = time.time()

    next_obs = envs.reset()
    next_image_obs = torch.tensor(next_obs["player_0"]["features"]).to(config["device"])
    unit_action_mask = torch.tensor(next_obs["player_0"]["invalid_unit_action_mask"]).to(config["device"])
    factory_action_mask = torch.tensor(next_obs["player_0"]["invalid_factory_action_mask"]).to(config["device"])    
    unit_mask = torch.tensor(next_obs["player_0"]["unit_mask"]).to(config["device"])
    factory_mask = torch.tensor(next_obs["player_0"]["factory_mask"]).to(config["device"])

    next_done = torch.zeros(config["parallel_envs"]).to(device)
    num_updates = config["total_timesteps"] // config["batch_size"]
    episode_lengths = []    
    episode_rewards = []
    KL_factor = config["KL_factor"]

    if config["KL_factor"] > 0 or config["teacher_path"] is None:
        #TODO: Find a better way to load the agent
        KL_loss_fn = torch.nn.KLDivLoss(reduce="batchmean", log_target=True)
        teacher = Agent({"device": config["device"],
                        "mode_or_sample": config["mode_or_sample"],
                        "mean_entropy": config["mean_entropy"],
                        "path": config["teacher_path"],
                        "actor_n_blocks": config["teacher_actor_n_blocks"],
                        "actor_intermediate_channels": config["teacher_actor_intermediate_channels"],
                        "critic_intermediate_channels": config["critic_intermediate_channels"],
                        "kernel_size": config["kernel_size"],
                        "critic_n_blocks": config["critic_n_blocks"]})


    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if config["KL_steps_to_anneal"] > 0:
            frac = 1.0 - max(0, global_step/config["KL_steps_to_anneal"])
            KL_factor = frac*config["KL_factor"]

        #envs.update_config(new_config)

        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["lr"]
            lrnow = max(5e-5, lrnow)
            optimizer.param_groups[0]["lr"] = lrnow

        if config["anneal_gamma_lambda"]:
            frac = (update - 1.0) / num_updates
            gamma_now = min(1, (1 - frac) * config["gamma"] + frac * 1.0)
            lambda_now = min(1, (1 - frac) * config["lmbda"] + frac * 1.0)
        else:
            gamma_now, lambda_now = config["gamma"], config["lmbda"]

        stepping_time = 0
        pre_and_post_step = time.time()
        with tqdm(total = config["num_steps_per_env"]*config["parallel_envs"], desc = "Playing", leave = False) as pbar:
            for step in range(0, config["num_steps_per_env"]):
                global_step += config["parallel_envs"]
                image_obs[step] = next_image_obs
                dones[step] = next_done
                unit_action_masks[step] = unit_action_mask
                factory_action_masks[step] = factory_action_mask
                unit_masks[step] = unit_mask
                factory_masks[step] = factory_mask

                state = {"features": next_image_obs,
                         "unit_mask": unit_mask,
                         "factory_mask": factory_mask,
                         "invalid_unit_action_mask": unit_action_mask,
                         "invalid_factory_action_mask": factory_action_mask}

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    factory_action, factory_action_logprob, _, _, \
                    light_unit_action, light_unit_action_logprob, _, _, \
                    heavy_unit_action,  heavy_unit_action_logprob,  _, _, \
                    value = agent.get_action_and_value(state)

                    opponent_factory_action, opponent_light_action, opponent_heavy_action, _ = opponent.get_action(next_obs["player_1"])
                    
                    values[step] = value.flatten()

                factory_actions[step] = factory_action
                factory_logprobs[step] = factory_action_logprob

                light_unit_actions[step] = light_unit_action
                light_unit_logprobs[step] = light_unit_action_logprob

                heavy_unit_actions[step] = heavy_unit_action
                heavy_unit_logprobs[step] = heavy_unit_action_logprob

                factory_action, light_unit_action, heavy_unit_action = factory_action.cpu().numpy(), light_unit_action.cpu().numpy(), heavy_unit_action.cpu().numpy()
                

                action = [{"player_0" : { "factory_action": factory_action[i],
                                        "light_unit_action" : light_unit_action[i],
                                        "heavy_unit_action": heavy_unit_action[i]
                                        },

                         "player_1" : { "factory_action": opponent_factory_action[i],
                                        "light_unit_action" : opponent_light_action[i],
                                        "heavy_unit_action" : opponent_heavy_action[i]}}
                        for i in range(config["parallel_envs"])] #NOTE: Atm just a copy
                

                single_stepping_time = time.time()
                next_obs, reward, done, info = envs.step(action)
                stepping_time += time.time() - single_stepping_time
                #Collect stats:
                for d, single_info in zip(done, info):
                    if d:
                        stat_collector.update(single_info["stats"])
                        episode_lengths.append(single_info["episode_length"])
                        episode_rewards.append(single_info["stats"]["total_episodic_reward"])

                next_image_obs = torch.tensor(next_obs["player_0"]["features"]).to(config["device"])
                unit_action_mask = torch.tensor(next_obs["player_0"]["invalid_unit_action_mask"]).to(config["device"])
                factory_action_mask = torch.tensor(next_obs["player_0"]["invalid_factory_action_mask"]).to(config["device"])
                unit_mask = torch.tensor(next_obs["player_0"]["unit_mask"]).to(config["device"])
                factory_mask = torch.tensor(next_obs["player_0"]["factory_mask"]).to(config["device"])

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.Tensor(done).to(device)
                global_games_played += next_done.sum()
                last_steps_played += config["parallel_envs"]
                steps_since_opponent_update += config["parallel_envs"]
                last_games_played += next_done.sum().item()

                pbar.update(config["parallel_envs"])

        pre_and_post_step = time.time()-pre_and_post_step
        training_time = time.time()
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_image_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config["num_steps_per_env"])):
                if t == config["num_steps_per_env"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma_now* nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma_now * lambda_now * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_image_obs = image_obs.reshape((-1,) + envs.observation_space["player_0"]["features"].shape[1:])

        b_unit_action_masks = unit_action_masks.reshape((-1,) + (config["map_size"], config["map_size"], UNIT_ACTION_IDXS))
        b_unit_masks = unit_masks.reshape((-1,) + (config["map_size"], config["map_size"]))

        b_light_unit_logprobs = light_unit_logprobs.reshape(-1)
        b_light_unit_actions = light_unit_actions.reshape((-1,) + (config["map_size"], config["map_size"]))

        b_heavy_unit_logprobs = heavy_unit_logprobs.reshape(-1)
        b_heavy_unit_actions = heavy_unit_actions.reshape((-1,) + (config["map_size"], config["map_size"]))

        b_factory_action_masks = factory_action_masks.reshape((-1,) + (config["map_size"], config["map_size"], FACTORY_ACTION_IDXS))
        b_factory_logprobs = factory_logprobs.reshape(-1)
        b_factory_actions = factory_actions.reshape((-1,) + (config["map_size"], config["map_size"]))
        b_factory_masks = factory_masks.reshape((-1,) + (config["map_size"], config["map_size"]))

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config["batch_size"])
        clipfracs = []
        #We only need to do this once
        if KL_factor > 0:
            with torch.no_grad():
                old_factory_dist, old_light_unit_dist, old_heavy_unit_dist, _ = teacher.get_dist_and_value(rotate_tensor(b_image_obs[mb_inds], epoch),
                                                                                                           rotate_tensor(b_unit_masks[mb_inds], epoch), 
                                                                                                           rotate_tensor(b_factory_masks[mb_inds], epoch))

        for epoch in range(config["epochs_per_batch"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["mini_batch_size"]):
                end = start + config["mini_batch_size"]
                mb_inds = b_inds[start:end]

                state = {"features": rotate_tensor(b_image_obs[mb_inds], epoch),
                         "unit_mask": rotate_tensor(b_unit_masks[mb_inds], epoch),
                         "factory_mask": rotate_tensor(b_factory_masks[mb_inds], epoch),
                         "invalid_unit_action_mask": rotate_tensor(b_unit_action_masks[mb_inds], epoch),
                         "invalid_factory_action_mask": rotate_tensor(b_factory_action_masks[mb_inds], epoch)}
                
                _, factory_newlogprob, factory_entropy, factory_dist, \
                _, light_unit_newlogprob, light_unit_entropy, light_unit_dist, \
                _,  heavy_unit_newlogprob,  heavy_unit_entropy, heavy_unit_dist, \
                newvalue = agent.get_action_and_value(state,
                                                        rotate_tensor(b_factory_actions[mb_inds], epoch),
                                                        rotate_tensor(b_light_unit_actions[mb_inds], epoch),
                                                        rotate_tensor(b_heavy_unit_actions[mb_inds], epoch),
                                                        return_dist = KL_factor > 0)
                
                mb_advantages = b_advantages[mb_inds]

                if config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                ##############LIGHT-UNITS###################
                logratio = light_unit_newlogprob - b_light_unit_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_light_unit = (-logratio).mean()
                    approx_kl_light_unit = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["eps_clip"]).float().mean().item()]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["eps_clip"], 1 + config["eps_clip"])
                pg_light_unit_loss = torch.max(pg_loss1, pg_loss2).mean()
                ######################################

                ##############HEAVY-UNITS###################
                logratio = heavy_unit_newlogprob - b_heavy_unit_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_heavy_unit = (-logratio).mean()
                    approx_kl_heavy_unit = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["eps_clip"]).float().mean().item()]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["eps_clip"], 1 + config["eps_clip"])
                pg_heavy_unit_loss = torch.max(pg_loss1, pg_loss2).mean()
                ######################################

                ############FACTORIES#################
                logratio = factory_newlogprob - b_factory_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_factory = (-logratio).mean()
                    approx_kl_factory = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["eps_clip"]).float().mean().item()]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["eps_clip"], 1 + config["eps_clip"])
                pg_factory_loss = torch.max(pg_loss1, pg_loss2).mean()
                ######################################

                approx_kl = (approx_kl_factory + approx_kl_light_unit + old_approx_kl_heavy_unit)/3
                old_approx_kl = (old_approx_kl_factory + approx_kl_light_unit + approx_kl_heavy_unit)/3

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

               
                entropy_loss = config["ent_coef"] * (factory_entropy.mean() + light_unit_entropy.mean() + heavy_unit_entropy.mean())/3
                loss = (pg_factory_loss + pg_light_unit_loss + pg_heavy_unit_loss) -  entropy_loss + v_loss*config["vf_coef"]

                if KL_factor > 0:
                    unit_KL_loss = KL_loss_fn(light_unit_dist.squeeze(), old_light_unit_dist.squeeze(), old_heavy_unit_dist.squeeze())
                    factory_KL_loss = KL_loss_fn(factory_dist.squeeze(), old_factory_dist.squeeze())

                    KL_loss = (factory_KL_loss + unit_KL_loss)*KL_factor
                    loss += KL_loss

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

        if steps_since_opponent_update > config["opponent_update_rate"]:
            #sampled_opponent = random.choice(possible_opponents)
            #opponent.load(sampled_opponent)
            opponent.load(agent.model.state_dict())
            steps_since_opponent_update = 0
            #if save_with_retry(agent, f"Opponents/{global_step}"):
            #    possible_opponents.append(f"Opponents/{global_step}")


        if last_steps_played > config["log_rate"]:
            #agent.save("most_recent.t")
            save_with_retry(agent, f"models/most_recent.t")

            categories = stat_collector.get_last_x(int(last_games_played))
            print(f"SPS: {int(last_steps_played / (time.time() - start_time))} Reward: {round(np.mean(episode_rewards).item(), 2)} Water: {round(np.mean(categories['main']['water_made']).item(), 0)} Games played: {int(global_games_played)} Update {update}/{num_updates}")
            if np.mean(episode_rewards) > higest_average:
                higest_average = np.mean(episode_rewards)
                agent.save(config["save_path"])
            if config["log_to_wb"]:
                log_dict = {}
                log_dict["Main/Average reward per step"] = np.mean(episode_rewards).item()
                log_dict["Main/Average episode length"] = np.mean(episode_lengths)
                log_dict["Main/Win rate"] = 100*(((np.mean(categories["rewards"]["win_reward"]))+1)/2)
                #log_dict["Main/Max episode length"] = np.max(episode_lengths)
                if KL_factor > 0:
                    log_dict["Losses/KL_loss"] = KL_loss.item()
                log_dict["Losses/value_loss"] = v_loss.item()
                log_dict["Losses/policy_loss"] = (pg_factory_loss + pg_light_unit_loss + pg_heavy_unit_loss).item()
                log_dict["Losses/entropy_loss"] = entropy_loss.item()
                log_dict["Losses/approx_kl"] = approx_kl.item()
                log_dict["Losses/clipfrac"] = np.mean(clipfracs)
                log_dict["Losses/explained_variance"] = explained_var
                log_dict["Charts/unit_entropy"] = (light_unit_entropy.mean() + heavy_unit_entropy.mean()).mean().item()
                log_dict["Charts/factory_entropy"] = factory_entropy.mean().item()
                log_dict["Charts/learning_rate"] = optimizer.param_groups[0]["lr"]
                if KL_factor > 0:
                    log_dict["Charts/KL_factory"] = KL_factor
                if config["anneal_gamma_lambda"]:
                    log_dict["Charts/gamma"] = gamma_now
                    log_dict["Charts/lambda"] = lambda_now
                log_dict["Charts/Steps per second"] = int(last_steps_played / (time.time() - start_time))
                log_dict["Charts/Portion of time as training"] = training_time/(training_time+pre_and_post_step+stepping_time)
                log_dict["Charts/Portion of time as misc"] = pre_and_post_step/(training_time+pre_and_post_step+stepping_time)
                log_dict["Charts/Portion of time in env.step "] = stepping_time/(training_time+pre_and_post_step+stepping_time)
                for category_name, category in categories.items():
                    for name, value in category.items():
                        log_dict[f"{category_name}/{name}"] = np.mean(value)

                logger.log(log_dict, global_step)
            last_steps_played = 0
            last_games_played = 0
            start_time = time.time()
            episode_rewards = []
            episode_lengths = []
    print("Finished training")
    envs.close()
