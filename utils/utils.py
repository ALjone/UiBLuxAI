import torch 
import yaml
import numpy as np
def load_config(change_dict = {}):
    config = yaml.safe_load(open("config.yml"))

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
    tile_locations = np.argwhere(tile_map == 1)
    
    tile_distances = np.mean(
        (tile_locations - np.array(unit_pos)), 1
    )
    # normalize the ice tile location
    closest_tile = (
        tile_locations[np.argmin(tile_distances)]
    )   
    return closest_tile 