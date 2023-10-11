from luxai_s2.env import LuxAI_S2
from agents.RL_agent import Agent
from agents.opponent import Opponent
from utils.visualization import animate, visualize_stats
from wrappers.observation_wrapper import StateSpaceVol2
from wrappers.other_wrappers import SinglePlayerEnv
from wrappers.reward_wrappers import IceRewardWrapper
from wrappers.action_wrapper import action_wrapper
from wrappers.observations_config import FACTORY_MAX_RES
from utils.utils import load_config
import torch
from utils.visualization import visualize_obs, visualize_network_output, visualize_critic_and_returns
from actions.actions import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS

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
    opponent_critic_output = []
    unit_logits = torch.zeros(UNIT_ACTION_IDXS)
    factory_logits = torch.zeros(FACTORY_ACTION_IDXS)
    lichen_player = []
    lichen_opponent = []
    water = []
    while not done:
        torch_state = {}
        for k, v in state["player_0"].items():
            torch_state[k] = torch.tensor(v)

        with torch.no_grad():
            unit_action, _, _, _, factory_action, _, _, _, _ = agent.get_action_and_value(torch_state)
            opponent_unit_actions, opponent_factory_action, opponent_state_val = opponent.get_action(state["player_1"])
            #opponent_unit_actions = torch.zeros((48, 48))
            #opponent_factory_action = torch.ones((48, 48))*3
        action = {  "player_0" : {"unit_action" : unit_action.squeeze(), "factory_action": factory_action.squeeze()},
                    "player_1" : {"unit_action" : opponent_unit_actions, "factory_action": opponent_factory_action}} #NOTE: Atm just a copy
        step += 1
        print(step)
        lichen_player.append(torch_state["features"][15].sum().item()*env.state.env_cfg.MAX_LICHEN_PER_TILE)
        lichen_opponent.append(torch_state["features"][16].sum().item()*env.state.env_cfg.MAX_LICHEN_PER_TILE)
        #water.append((FACTORY_MAX_RES*(torch_state["features"][10]*torch_state["features"][2]/9)).sum().item())
        w = torch.tensor(0.0)

        for factory in env.get_state().get_obs()["factories"]["player_0"].values():
            w += factory["cargo"]["water"]
        water.append(w.item())

        """with torch.no_grad():
            unit_action_logits, factory_action_logits = agent.get_action_probs(torch_state)
            unit_probs = torch.nn.functional.softmax(unit_action_logits, -1)
            factory_probs = torch.nn.functional.softmax(factory_action_logits, -1)
        for unit_id, unit in env.get_state().units["player_0"].items():
            x, y = unit.pos.x, unit.pos.y
            print(f"{unit_id}: {[round(x.item(), 4) for x in list(unit_probs[x, y])]}")
            unit_logits += unit_probs[x, y]
        for factory_id, factory in env.get_state().factories["player_0"].items():
            x, y = factory.pos.x, factory.pos.y
            print(f"{factory_id}: {[round(x.item(), 4) for x in list(factory_probs[x, y])]}")
            factory_logits += factory_probs[x, y]"""
        
        if make_network_gif:    
            with torch.no_grad():
                unit_action_logits, factory_action_logits = agent.get_action_probs(torch_state)
            img = visualize_network_output(unit_action_logits.cpu().numpy(), factory_action_logits.cpu().numpy(), torch_state["unit_mask"].numpy(), torch_state["factory_mask"].numpy(), step)
            output_imgs.append(img)

        if make_obs_gif:
            img = visualize_obs(torch_state, step)
            obs_imgs.append(img)

        critic_output.append(agent.get_value(torch.tensor(state["player_0"]["features"])).detach().cpu().numpy().squeeze())
        opponent_critic_output.append(opponent_state_val)
        state, r, done, _ = env.step(action)

        reward += r
        rewards.append(r)

        

        if make_gif:
            map_imgs += [env.render("rgb_array", width=640, height=640)]

        if step >= step_num:
            break
    
    print("Avg logits unit:", [round(x.item(), 4) for x in list(unit_logits/torch.sum(unit_logits))])
    print("Avg factory unit:", [round(x.item(), 4) for x in list(factory_logits/torch.sum(factory_logits))])

    print("Making gif")
    if make_gif:
        animate(map_imgs, "gifs/map.gif")
    if make_obs_gif:
        animate(obs_imgs, "gifs/observation.gif")
    if make_network_gif:
        animate(output_imgs, "gifs/network_output.gif")

    visualize_critic_and_returns(calculate_return(rewards, gamma = gamma), critic_output, opponent_critic_output, rewards)
    visualize_stats(lichen_player, lichen_opponent, water)
    print("Got a total of:", round(reward, 1), "reward")

def run(config):
    #Always run visualization on CPU
    config["device"] = torch.device("cpu")
    env = LuxAI_S2(verbose=1, collect_stats=True, map_size = config["map_size"], MIN_FACTORIES = config["n_factories_min"], MAX_FACTORIES = config["n_factories_max"], validate_action_space = True)
    env.reset() #NOTE: Reset here to initialize stats
    env = SinglePlayerEnv(env, config)
    env = StateSpaceVol2(env, config)
    env = IceRewardWrapper(env, config)
    env.reset()
    env = action_wrapper(env)
    env.reset()
    if config["path"] is None:
        config["path"] = "models/most_recent.t"#"with_fixed_reward_8x64.t"
    config["reset_critic"] = False
    #TODO: Also log opponent critic
    agent = Agent(config)
    opponent = Opponent(config)
    opponent.load(agent.model.state_dict())
    #play_episode(agent, opponent, env, config["gamma"], True, True, True, 100)
    #play_episode(agent, opponent, env, config["gamma"], True, False, False, 1000)
    play_episode(agent, opponent, env, config["gamma"], False, False, False, 1000)

if __name__ == "__main__":
    config = load_config()
    config["mode_or_sample"] = "sample"
    run(config)