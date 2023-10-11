
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from agents.opponent import Opponent
from utils.stat_collector import StatCollector
from utils.utils import load_config, save_with_retry
import time
import gym
from tqdm import tqdm


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

    device = config["device"]
    
    print("NOTE: Currently running without power cost.")    
    print("Running with mini batch size:", config["mini_batch_size"], "Batch size:", config["batch_size"])

    if not (config["mini_batch_size"] & (config["mini_batch_size"]-1) == 0):
        print("NBNBNBNB!!! Running with a non-power-of-two batch size!")
    print("Will run:", config["total_timesteps"] // config["batch_size"], "updates.")

    assert config["path"] is not None, "There's no point in running only critic when there is no model"

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config, i) for i in range(config["parallel_envs"])]
    )

    agent = Agent(config)

    if config["self_play"]:
        agent.save("Opponents/start.t")
        opponent = Opponent(config)
        opponent.load("Opponents/start.t")


    optimizer = optim.Adam(agent.model.parameters(), lr=config["lr"])

    image_obs = torch.zeros((config["num_steps_per_env"], config["parallel_envs"]) + envs.observation_space["player_0"]["features"].shape[1:]).to(device)

    rewards = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    dones = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)
    values = torch.zeros((config["num_steps_per_env"], config["parallel_envs"])).to(device)

    # TRY NOT TO MODIFY: start the game
    last_steps_played = 0

    start_time = time.time()

    next_obs = envs.reset()
    next_image_obs = torch.tensor(next_obs["player_0"]["features"]).to(config["device"])
    unit_action_mask = torch.tensor(next_obs["player_0"]["invalid_unit_action_mask"]).to(config["device"])
    factory_action_mask = torch.tensor(next_obs["player_0"]["invalid_factory_action_mask"]).to(config["device"])    
    unit_mask = torch.tensor(next_obs["player_0"]["unit_mask"]).to(config["device"])
    factory_mask = torch.tensor(next_obs["player_0"]["factory_mask"]).to(config["device"])

    next_done = torch.zeros(config["parallel_envs"]).to(device)
    num_updates = config["total_timesteps"] // config["batch_size"]

    KL_factor = config["KL_factor"]

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.

        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["lr"]
            lrnow = max(5e-5, lrnow)
            optimizer.param_groups[0]["lr"] = lrnow

        stepping_time = 0
        with tqdm(total = config["num_steps_per_env"]*config["parallel_envs"], desc = "Playing", leave = False, disable = False) as pbar:
            for step in range(0, config["num_steps_per_env"]):
                image_obs[step] = next_image_obs
                dones[step] = next_done

                state = {"features": next_image_obs,
                         "unit_mask": unit_mask,
                         "factory_mask": factory_mask,
                         "invalid_unit_action_mask": unit_action_mask,
                         "invalid_factory_action_mask": factory_action_mask}

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    unit_action, _, _, _, factory_action, _, _, _, value = agent.get_action_and_value(state, return_dist = False)
                    if config["self_play"]:
                        opponent_action, opponent_factory_action = opponent.get_action(next_obs["player_1"])
                    else:
                        opponent_action = torch.zeros((16, 48, 48)).to(device)
                        opponent_factory_action = torch.ones((16, 48, 48)).to(device)*3
                    
                    values[step] = value.flatten()


                unit_action, factory_action = unit_action.cpu().numpy(), factory_action.cpu().numpy()
                

                action = [{"player_0" : {"unit_action" : unit_action[i], "factory_action": factory_action[i]},
                          "player_1" : {"unit_action" : opponent_action[i], "factory_action": opponent_factory_action[i]}} for i in range(config["parallel_envs"])] #NOTE: Atm just a copy
                next_obs, reward, done, info = envs.step(action)
                #Collect stats:

                next_image_obs = torch.tensor(next_obs["player_0"]["features"]).to(config["device"])
                unit_action_mask = torch.tensor(next_obs["player_0"]["invalid_unit_action_mask"]).to(config["device"])
                factory_action_mask = torch.tensor(next_obs["player_0"]["invalid_factory_action_mask"]).to(config["device"])
                unit_mask = torch.tensor(next_obs["player_0"]["unit_mask"]).to(config["device"])
                factory_mask = torch.tensor(next_obs["player_0"]["factory_mask"]).to(config["device"])

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.Tensor(done).to(device)
                last_steps_played += config["parallel_envs"]

                pbar.update(config["parallel_envs"])


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
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["lmbda"] * nextnonterminal * lastgaelam
            returns = advantages + values


        # Apply eligibility traces to the returns
        #returns *= eligibility_traces

        # Flatten the batch
        b_image_obs = image_obs.reshape((-1,) + envs.observation_space["player_0"]["features"].shape[1:])
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config["batch_size"])
        for epoch in range(config["critic_epochs_per_batch"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["mini_batch_size"]):
                end = start + config["mini_batch_size"]
                mb_inds = b_inds[start:end]

                newvalue = agent.get_value(b_image_obs[mb_inds])
                newvalue = newvalue.view(-1)
                
                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                #v_loss = torch.abs(newvalue-b_returns[mb_inds]).mean()

                optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), config["max_grad_norm"])
                optimizer.step()


            
        save_with_retry(agent, f"models/most_recent.t")

        print(f"SPS: {int(last_steps_played / (time.time() - start_time))} Loss: {round(v_loss.item(), 6)} LR: {round(lrnow, 5)} Update {update}/{num_updates}")
        last_steps_played = 0
        start_time = time.time()
    print("Finished training")
    envs.close()
