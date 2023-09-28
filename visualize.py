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
from utils.visualization import visualize_obs, visualize_network_output, visualize_critic_and_returns

def calculate_return(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return returns[::-1]

def play_episode(agent: Agent, opponent: Opponent, env: LuxAI_S2, gamma, make_gif = True, make_obs_gif = False, make_network_gif = False, step_num = 300):
    state = env.reset()
    agent.model.eval()
    step = 0
    map_imgs = []
    obs_imgs = []
    output_imgs = []
    done = False
    rewards = []
    reward = 0
    critic_output = []
    while not done:
        torch_state = {}
        for k, v in state["player_0"].items():
            torch_state[k] = torch.tensor(v)

        with torch.no_grad():
            unit_action, _, _, factory_action, _, _, _ = agent.get_action_and_value(torch_state)
            #opponent_unit_actions, opponent_factory_action = opponent.get_action(state["player_1"])
            opponent_unit_actions = torch.zeros((48, 48))
            opponent_factory_action = torch.ones((48, 48))*3
        action = {  "player_0" : {"unit_action" : unit_action.squeeze(), "factory_action": factory_action.squeeze()},
                    "player_1" : {"unit_action" : opponent_unit_actions, "factory_action": opponent_factory_action}} #NOTE: Atm just a copy
        step += 1
        print(step)


        if make_network_gif:
            with torch.no_grad():
                unit_action, factory_action = agent.get_action_probs(torch_state)
            img = visualize_network_output(unit_action.cpu().numpy(), factory_action.cpu().numpy(), torch_state["unit_mask"].numpy(), torch_state["factory_mask"].numpy(), step)
            output_imgs.append(img)

        if make_obs_gif:
            img = visualize_obs(torch_state, step)
            obs_imgs.append(img)

        critic_output.append(agent.get_value(torch.tensor(state["player_0"]["features"])).detach().cpu().numpy().squeeze())
        state, r, done, _ = env.step(action)

        reward += r
        rewards.append(r)

        

        if make_gif:
            map_imgs += [env.render("rgb_array", width=640, height=640)]

        if step >= step_num:
            break

    print("Making gif")
    if make_gif:
        animate(map_imgs, "map.gif")
    if make_obs_gif:
        animate(obs_imgs, "observation.gif")
    if make_network_gif:
        animate(output_imgs, "network_output.gif")

    visualize_critic_and_returns(calculate_return(rewards, gamma = gamma), critic_output, rewards)
    print("Got a total of:", round(reward, 1), "reward")

def run(config):
    #Always run visualization on CPU
    config["device"] = torch.device("cpu")
    #config["map_size"] = 48
    #config["n_factories"] = 5
    env = LuxAI_S2(verbose=1, collect_stats=True, map_size = config["map_size"], MIN_FACTORIES = config["n_factories_min"], MAX_FACTORIES = config["n_factories_max"], validate_action_space = True)
    env.reset() #NOTE: Reset here to initialize stats
    env = SinglePlayerEnv(env, config)
    env = StateSpaceVol2(env, config)
    env = IceRewardWrapper(env, config)
    env.reset()
    env = action_wrapper(env)
    env.reset()
    config["path"] = "most_recent.t"
    agent = Agent(config)
    opponent = Opponent(config)
    opponent.load(agent.model.state_dict())
    #play_episode(agent, opponent, env, config["gamma"], True, True, True, 100)
    play_episode(agent, opponent, env, config["gamma"], True, False, False, 1000)


if __name__ == "__main__":
    config = load_config()
    config["mode_or_sample"] = "mode"
    run(config)