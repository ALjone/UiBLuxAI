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

    if config["load_path"] == "None":
        config["load_path"] = None

    return config