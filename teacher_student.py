
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from agents.opponent import Opponent
from utils.utils import load_config
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
import os
os.environ["WANDB_SILENT"] = "true"

def make_env(config):
    def thunk():
        env = LuxAI_S2(verbose=0, collect_stats=True, map_size = config["map_size"], MIN_FACTORIES = config["n_factories"], MAX_FACTORIES = config["n_factories"], validate_action_space = False)
        env.reset() #NOTE: Reset here to initialize stats
        env = SinglePlayerEnv(env, config)
        env = StateSpaceVol2(env, config)
        env.reset()
        env = action_wrapper(env)
        env.reset()
        return env

    return thunk


if __name__ == "__main__":
    config = load_config(path = "teacher_student_config.yml")
    run_name = str(time.time())
    device = config["device"]
    
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config) for _ in range(config["parallel_envs"])]
    )

    agent = Agent(config)
    agent.save("Opponents/start.t")
    opponent = Opponent(config)
    opponent.load(agent.model.state_dict())

    student = Agent({"device": config["device"],
                     "mode_or_sample": config["mode_or_sample"],
                     "path": None,
                     "actor_n_blocks": config["student_actor_n_blocks"],
                     "actor_intermediate_channels": config["student_actor_intermediate_channels"],
                     "actor_use_batch_norm": config["student_actor_use_batch_norm"]})

    optimizer = optim.Adam(student.model.parameters(), lr=config["lr"], eps=1e-5)

    # ALGO Logic: Storage setup
    image_obs = torch.zeros((config["batch_size"], config["parallel_envs"]) + envs.observation_space["player_0"]["features"].shape[1:]).to(device)

    unit_action_masks = torch.zeros((config["batch_size"], config["parallel_envs"]) + (config["map_size"], config["map_size"], UNIT_ACTION_IDXS)).to(device)
    unit_actions = torch.zeros((config["batch_size"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    unit_logprobs = torch.zeros((config["batch_size"], config["parallel_envs"])).to(device)
    unit_masks = torch.zeros((config["batch_size"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)

    factory_action_masks = torch.zeros((config["batch_size"], config["parallel_envs"]) + (config["map_size"], config["map_size"], FACTORY_ACTION_IDXS)).to(device)
    factory_actions = torch.zeros((config["batch_size"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)
    factory_logprobs = torch.zeros((config["batch_size"], config["parallel_envs"])).to(device)
    factory_masks = torch.zeros((config["batch_size"], config["parallel_envs"]) + (config["map_size"], config["map_size"])).to(device)

    values = torch.zeros((config["batch_size"], config["parallel_envs"])).to(device)

    KL_loss = torch.nn.KLDivLoss(reduce="batchmean", log_target=True)
    value_loss = torch.nn.MSELoss()

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    global_games_played = 0
    last_steps_played = 0
    last_games_played = 0
    higest_average = 0


    next_obs = envs.reset()
    next_image_obs = torch.tensor(next_obs["player_0"]["features"]).to(config["device"])
    unit_action_mask = torch.tensor(next_obs["player_0"]["invalid_unit_action_mask"]).to(config["device"])
    factory_action_mask = torch.tensor(next_obs["player_0"]["invalid_factory_action_mask"]).to(config["device"])    
    unit_mask = torch.tensor(next_obs["player_0"]["unit_mask"]).to(config["device"])
    factory_mask = torch.tensor(next_obs["player_0"]["factory_mask"]).to(config["device"])

    next_done = torch.zeros(config["parallel_envs"]).to(device)
    num_updates = config["max_episodes"] // config["batch_size"]

    num_train_updates = 0


    for update in range(1, num_updates + 1):
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["lr"]
            optimizer.param_groups[0]["lr"] = lrnow
        with tqdm(total = config["num_steps_per_env"]*config["parallel_envs"], desc = "Playing", leave = False) as pbar:
            for step in range(0, config["num_steps_per_env"]):
                global_step += config["parallel_envs"]
                image_obs[step] = next_image_obs

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
                    #opponent_action, opponent_factory_action = opponent.get_action(next_obs["player_1"])
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


                next_image_obs = torch.tensor(next_obs["player_0"]["features"]).to(config["device"])
                unit_action_mask = torch.tensor(next_obs["player_0"]["invalid_unit_action_mask"]).to(config["device"])
                factory_action_mask = torch.tensor(next_obs["player_0"]["invalid_factory_action_mask"]).to(config["device"])
                unit_mask = torch.tensor(next_obs["player_0"]["unit_mask"]).to(config["device"])
                factory_mask = torch.tensor(next_obs["player_0"]["factory_mask"]).to(config["device"])

                next_done = torch.Tensor(done).to(device)
                global_games_played += next_done.sum()
                last_steps_played += config["parallel_envs"]
                last_games_played += next_done.sum().item()

                pbar.update(config["parallel_envs"])


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

        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config["batch_size"])
        
        unit_losses = []
        factory_losses = []
        value_losses = []

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


                unit_dist, factory_dist, newvalue = student.get_dist_and_value(b_image_obs[mb_inds], b_unit_masks[mb_inds], b_factory_masks[mb_inds])

                with torch.no_grad():
                    old_unit_dist, old_factory_dist, old_value = agent.get_dist_and_value(b_image_obs[mb_inds], b_unit_masks[mb_inds], b_factory_masks[mb_inds])
                

                unit_loss = KL_loss(unit_dist.squeeze(), old_unit_dist.squeeze())
                factory_loss = KL_loss(factory_dist.squeeze(), old_factory_dist.squeeze())

                critic_loss = value_loss(newvalue.squeeze(), old_value.squeeze())

                loss = unit_loss + critic_loss + factory_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                unit_losses.append(unit_loss.item())
                factory_losses.append(factory_loss.item())
                value_losses.append(critic_loss.item())


        if last_steps_played > config["log_rate"]:
            student.save(config["student_save_path"])
            print(f"Epoch: {num_train_updates}, Unit loss: {round(np.mean(unit_losses).item(), 7)}, Factory loss: {round(np.mean(factory_losses).item(), 7)}, Critic loss: {round(np.sum(value_losses).item(), 4)}")
            num_train_updates += 1
            last_steps_played = 0
            
    print("Finished training")
    envs.close()
