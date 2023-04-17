from luxai_s2.env import LuxAI_S2
from agents.RL_agent import Agent
from agents.opponent import Opponent
from utils.visualization import animate
from wrappers.observation_wrapper import StateSpaceVol2
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from wrappers.action_wrapper import action_wrapper
from utils.utils import load_config
import torch
from utils.visualization import visualize_obs, visualize_network_output
import numpy as np

def play_episode(agent: Agent, opponent: Opponent, env: LuxAI_S2, make_gif = True, make_obs_gif = False, make_network_gif = False):
    state = env.reset()
    agent.model.eval()
    step = 0
    map_imgs = []
    obs_imgs = []
    output_imgs = []
    done = False
    while not done:
        with torch.no_grad():
            unit_action, factory_action = agent.get_action(state["player_0"])
            opponent_unit_actions, opponent_factory_action = opponent.get_action(state["player_1"])
        action = {"player_0" : {"unit_action" : unit_action, "factory_action": factory_action},
                          "player_1" : {"unit_action" : opponent_unit_actions, "factory_action": opponent_factory_action}} #NOTE: Atm just a copy
        step += 1
        print(step)
        #max_ = np.max(state["player_0"]["features"])
        #if max_ > 2:    
        #    print("Max feature:", max_)

        if make_network_gif:
            with torch.no_grad():
                unit_action, factory_action = agent.get_action_probs(state["player_0"])
            img = visualize_network_output(unit_action.cpu().numpy(), factory_action.cpu().numpy(), state["player_0"]["unit_mask"], state["player_0"]["factory_mask"], step)
            output_imgs.append(img)

        if make_obs_gif:
            img = visualize_obs(state["player_0"], step)
            obs_imgs.append(img)
        state, _, done, _ = env.step(action)

        if make_gif:
            map_imgs += [env.render("rgb_array", width=640, height=640)]

        if step > 300:
            break

    print("Making gif")
    if make_gif:
        animate(map_imgs, "map.gif")
    if make_obs_gif:
        animate(obs_imgs, "observation.gif")
    if make_network_gif:
        animate(output_imgs, "network_output.gif")

def run(config):
    #Always run visualization on CPU
    config["device"] = torch.device("cpu")
    env = LuxAI_S2(verbose=1, collect_stats=True, map_size = config["map_size"], MIN_FACTORIES = config["n_factories"], MAX_FACTORIES = config["n_factories"], validate_action_space = True)
    env.reset() #NOTE: Reset here to initialize stats
    env = SinglePlayerEnv(env, config)
    env = StateSpaceVol2(env, config)
    env = IceRewardWrapper(env, config)
    env.reset()
    env = action_wrapper(env)
    env.reset()
    agent = Agent(config)
    opponent = Opponent(config)
    opponent.load(agent.model.state_dict())
    play_episode(agent, opponent, env, True, False, True)


if __name__ == "__main__":
    config = load_config()
    config["mode_or_sample"] = "sample"
    run(config)