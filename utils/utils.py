import torch 
import yaml
import numpy as np
import time

def load_config(change_dict = {}, path = "config.yml"):
    config = yaml.safe_load(open(path))

    config["batch_size"] = config["parallel_envs"]*config["num_steps_per_env"]
    config["mini_batch_size"] = int(config["batch_size"] // config["num_minibatches"])

    if not (config["mini_batch_size"] & (config["mini_batch_size"]-1) == 0):
        print("Running with a non-power-of-two batch size!")

    
    if not config["device"].lower() in ["cpu", "cuda", "mps"]:
        raise ValueError("Expected device in Config to be either 'CPU' or 'CUDA', but found:", config["device"])

    if torch.cuda.is_available() and config["device"] == "cuda":
        config["device"] = torch.device("cuda")
    elif config["device"] == "mps":
        config["device"] = torch.device("mps")#torch.device("mps")
    else:
        config["device"] = torch.device("cpu")

    for key, val in change_dict.items():
        config[key] = val

    if config["path"] == "None":
        config["path"] = None

    return config

def save_with_retry(agent, path, retries=3, delay=2):
    for _ in range(retries):
        try:
            agent.save(path)
            return
        except RuntimeError as e:
            #print(f"Save failed: {e}, retrying...")
            time.sleep(delay)


def formate_time(seconds):
    seconds = int(seconds)
    #https://stackoverflow.com/a/775075
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if m == 0 and h == 0:
        return f'{int(s)} seconds' 
    if h == 0: 
        return f'{int(m)} minutes and {int(s)} seconds' 
    else:
        return f'{int(h)} hours, {int(m)} minutes and {int(s)} seconds'

def find_closest_tile(tile_map, unit_pos): 
    tile_locations = np.argwhere(tile_map > 0)
    
    tile_distances = np.mean(
        np.abs(tile_locations - np.array(unit_pos)), 1
    )
    if len(tile_distances) == 0:
        return (0, 0) #TODO: Default case, shady, but circumvents a bug
    # normalize the ice tile location
    closest_tile = (
        tile_locations[np.argmin(tile_distances)]
    )   

    return closest_tile 