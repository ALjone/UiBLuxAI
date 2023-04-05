from luxai_s2.env import LuxAI_S2
from wrappers.other_wrappers import SinglePlayerEnv
from utils.utils import load_config
import torch
import matplotlib.pyplot as plt
def play_episode(env: LuxAI_S2):
    i = "No"
    while True:
        env.reset()
        
        img = env.render("rgb_array", width=640, height=640)
        plt.imshow(img)
        plt.show()


def run(config):
    #Always run visualization on CPU
    config["device"] = torch.device("cpu")
    env = LuxAI_S2(verbose=1, collect_stats=True, MIN_FACTORIES = config["n_factories"], MAX_FACTORIES = config["n_factories"], validate_action_space = True)
    env.reset() #NOTE: Reset here to initialize stats
    env = SinglePlayerEnv(env, config)
    
    play_episode(env)


if __name__ == "__main__":
    config = load_config()
    run(config)