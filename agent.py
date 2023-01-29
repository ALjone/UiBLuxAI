from lux.kit import obs_to_game_state, EnvConfig
from lux.utils import my_turn_to_place_factory
import numpy as np
import torch
from network.actor import actor
from utils.utils import outputs_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS, unit_output_to_actions, self_destruct, unit_idx_to_action
from ppo import PPO

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig, device = torch.device("cuda"), path = None) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.device = device

        self.unit_actions_per_cell = UNIT_ACTION_IDXS
        self.factory_actions_per_cell = FACTORY_ACTION_IDXS

        self.PPO = PPO(self.unit_actions_per_cell, 3e-4, 3e-4, 0.99, 10, 0.1, device)

        if path is not None:
            self.PPO.load(path)
            print("Loaded model successfully")

        self.actions = [0]*UNIT_ACTION_IDXS

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, obs, remainingOverageTime: int = 60):
        image = obs["image_features"].to(self.device)
        units = obs["unit_to_id"]
        factories = obs["factory_to_id"]

        action = self.PPO.select_action(image).item()

        self.actions[action] += 1

        unit_actions = {}
        first = True
        #Such a bad way to do this
        for unit in units:
            if first:
                unit_actions[unit["unit_id"]] = unit_idx_to_action(action)
                first = False
            else:
                unit_actions[unit["unit_id"]] = self_destruct()
        
        
                
        if len(obs["unit_to_id"]) == 0:
            for factory in factories:
                if factory["cargo"]["metal"] >= 100:
                    unit_actions[factory["unit_id"]] = 1
                    break
        #NOTE How actions are formatted
        # a[0] (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        return unit_actions