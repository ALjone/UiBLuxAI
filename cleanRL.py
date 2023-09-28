
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from agents.opponent import Opponent
from utils.stat_collector import StatCollector
from utils.utils import load_config, save_with_retry
import time
import gym
from tqdm import tqdm

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
    print("Running with mini batch size:", config["mini_batch_size"])


    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config, i) for i in range(config["parallel_envs"])]
    )

    agent = Agent(config)
    assert config["self_play"] == False, "Self play???"
    if config["self_play"]:
        agent.save("Opponents/start.t")
        opponent = Opponent(config)
        opponent.load("Opponents/start.t")
    optimizer = optim.Adam(agent.model.parameters(), lr=config["lr"], eps=1e-5)

    image_obs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + envs.observation_space["player_0"]["features"].shape[1:]).to(device)

    #Unit
    unit_action_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"], UNIT_ACTION_IDXS)).to(device)
    unit_actions = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    unit_logprobs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    unit_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)

    #Factory
    factory_action_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"], FACTORY_ACTION_IDXS)).to(device)
    factory_actions = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    factory_logprobs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    factory_masks = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)

    rewards = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    dones = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    values = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    global_games_played = 0
    last_steps_played = 0
    last_games_played = 0
    higest_average = 0

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

    if config["KL_factor"] > 0:
        KL_loss = torch.nn.KLDivLoss(reduce="batchmean", log_target=True)
        teacher = Agent({"device": config["device"],
                        "mode_or_sample": config["mode_or_sample"],
                        "path": config["teacher_path"],
                        "actor_n_blocks": config["teacher_actor_n_blocks"],
                        "actor_intermediate_channels": config["teacher_actor_intermediate_channels"],
                        "actor_use_batch_norm": config["teacher_actor_use_batch_norm"]})


    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        frac = 1.0 - (global_step/config["KL_steps_to_anneal"])
        KL_factor = frac*config["KL_factor"]
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["lr"]
            optimizer.param_groups[0]["lr"] = lrnow

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
                    unit_action, unit_logprob, _, factory_action, factory_logprob, _, value = agent.get_action_and_value(state)
                    if config["self_play"]:
                        opponent_action, opponent_factory_action = opponent.get_action(next_obs["player_1"])
                    else:
                        opponent_action = torch.zeros((16, 48, 48)).to(device)
                        opponent_factory_action = torch.ones((16, 48, 48)).to(device)*3
                    
                    values[step] = value.flatten()

                unit_actions[step] = unit_action
                unit_logprobs[step] = unit_logprob
                factory_actions[step] = factory_action
                factory_logprobs[step] = factory_logprob

                unit_action, factory_action = unit_action.cpu().numpy(), factory_action.cpu().numpy()
                

                action = [{"player_0" : {"unit_action" : unit_action[i], "factory_action": factory_action[i]},
                          "player_1" : {"unit_action" : opponent_action[i], "factory_action": opponent_factory_action[i]}} for i in range(config["parallel_envs"])] #NOTE: Atm just a copy
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
                if t == config["num_steps_per_env"] - 1: #TODO: This looks a bit suspicious
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["lmbda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_image_obs = image_obs.reshape((-1,) + envs.observation_space["player_0"]["features"].shape[1:])

        b_unit_action_masks = unit_action_masks.reshape((-1,) + (config["map_size"], config["map_size"], UNIT_ACTION_IDXS))
        b_unit_logprobs = unit_logprobs.reshape(-1)
        b_unit_actions = unit_actions.reshape((-1,) + (config["map_size"], config["map_size"]))
        b_unit_masks = unit_masks.reshape((-1,) + (config["map_size"], config["map_size"]))

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
        for epoch in range(config["epochs_per_batch"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["mini_batch_size"]):
                end = start + config["mini_batch_size"]
                mb_inds = b_inds[start:end]

                state = {"features": b_image_obs[mb_inds],
                         "unit_mask": b_unit_masks[mb_inds],
                         "factory_mask": b_factory_masks[mb_inds],
                         "invalid_unit_action_mask": b_unit_action_masks[mb_inds],
                         "invalid_factory_action_mask": b_factory_action_masks[mb_inds]}
                _, unit_newlogprob, unit_entropy, _, factory_newlogprob, factory_entropy, newvalue = agent.get_action_and_value(state, b_unit_actions[mb_inds], b_factory_actions[mb_inds])
                mb_advantages = b_advantages[mb_inds]

                if config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                ##############UNITS###################
                logratio = unit_newlogprob - b_unit_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl_unit = (-logratio).mean()
                    approx_kl_unit = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["eps_clip"]).float().mean().item()]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["eps_clip"], 1 + config["eps_clip"])
                pg_unit_loss = torch.max(pg_loss1, pg_loss2).mean()
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

                approx_kl = approx_kl_unit+approx_kl_factory
                old_approx_kl = old_approx_kl_unit + old_approx_kl_factory

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


                entropy_loss = unit_entropy.mean()+factory_entropy.mean()
                loss = (pg_unit_loss + pg_factory_loss) - config["ent_coef"] * entropy_loss + v_loss * config["vf_coef"]

                if KL_factor > 0:
                    #Slow but who cares
                    unit_dist, factory_dist, _ = agent.get_dist_and_value(b_image_obs[mb_inds], b_unit_masks[mb_inds], b_factory_masks[mb_inds])

                    with torch.no_grad():
                        old_unit_dist, old_factory_dist, _ = teacher.get_dist_and_value(b_image_obs[mb_inds], b_unit_masks[mb_inds], b_factory_masks[mb_inds])
                    

                    unit_KL_loss = KL_loss(unit_dist.squeeze(), old_unit_dist.squeeze())
                    factory_KL_loss = KL_loss(factory_dist.squeeze(), old_factory_dist.squeeze())

                    loss += (unit_KL_loss + factory_KL_loss)*KL_factor

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
            if config["self_play"]:
                #save_with_retry(agent, f"Opponents/newest.t")
                opponent.load(agent.model.state_dict())
            categories = stat_collector.get_last_x(int(last_games_played))
            print(f"SPS: {int(last_steps_played / (time.time() - start_time))} Reward: {round(np.mean(episode_rewards).item(), 2)} Water: {round(np.mean(categories['main']['water_made']).item(), 0)} Games played: {int(global_games_played)}")
            if np.mean(episode_rewards) > higest_average:
                higest_average = np.mean(episode_rewards)
                agent.save(config["save_path"])
            if config["log_to_wb"]:
                log_dict = {}
                log_dict["Main/Average reward per step"] = np.mean(episode_rewards).item()
                log_dict["Main/Average episode length"] = np.mean(episode_lengths)
                #log_dict["Main/Max episode length"] = np.max(episode_lengths)
                if KL_factor > 0:
                    log_dict["Losses/Student_Teacher_KL"] = (unit_KL_loss+factory_KL_loss).item()
                log_dict["Losses/value_loss"] = v_loss.item()
                log_dict["Losses/policy_loss"] = (pg_unit_loss).item()
                log_dict["Losses/entropy"] = entropy_loss.item()
                log_dict["Losses/old_approx_kl"] = old_approx_kl.item()
                log_dict["Losses/approx_kl"] = approx_kl.item()
                log_dict["Losses/clipfrac"] = np.mean(clipfracs)
                log_dict["Losses/explained_variance"] = explained_var
                log_dict["Charts/learning_rate"] = optimizer.param_groups[0]["lr"]
                log_dict["Charts/KL_factory"] = KL_factor
                log_dict["Charts/Steps per second"] = int(last_steps_played / (time.time() - start_time))
                log_dict["Charts/Portion of time as training"] = training_time/(training_time+pre_and_post_step+stepping_time)
                log_dict["Charts/Portion of time as misc"] = pre_and_post_step/(training_time+pre_and_post_step+stepping_time)
                log_dict["Charts/Portion of time in env.step "] = stepping_time/(training_time+pre_and_post_step+stepping_time)
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
