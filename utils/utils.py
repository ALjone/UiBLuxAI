import torch 
import yaml
def load_config(change_dict = {}):
    config = yaml.safe_load(open("config.yml"))
    for key, val in change_dict.items():
        config[key] = val

    if config["device"].lower() in ["cpu", "cuda"]:
        config["device"] = torch.device(config["device"].lower())
    else:
        raise ValueError("Expected device in Config to be either 'CPU' or 'CUDA', but found:", config["device"])

    if config["path"] == "None":
        config["path"] = None

    return config


def formate_time(seconds):
    #https://stackoverflow.com/a/775075
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if m == 0 and h == 0:
        return f'{int(s)} seconds' 
    if h == 0: 
        return f'{int(m)} minutes and {int(s)} seconds' 
    else:
        return f'{int(h)} hours, {int(m)} minutes and {int(s)} seconds'
