from lux.kit import obs_to_game_state, EnvConfig
from lux.utils import my_turn_to_place_factory
import numpy as np
import scipy
from actions.idx_to_lux_move import outputs_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from ppo import PPO
import torch


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig, config) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg
        self.device = config["device"]

        self.unit_actions_per_cell = UNIT_ACTION_IDXS
        self.factory_actions_per_cell = FACTORY_ACTION_IDXS


        self.PPO = PPO(self.unit_actions_per_cell, self.factory_actions_per_cell, config)

        if config["path"] is not None:
            self.PPO.load(config["path"])
            print("Successfully loaded model")

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        obs = obs[1][self.player]
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
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))

                map = np.zeros((48, 48, 3))
                map[:, :, 0] = np.array(
                    obs["board"]["rubble"])/np.linalg.norm(np.array(obs["board"]["rubble"]))
                map[:, :, 1] = np.array(
                    obs["board"]["ore"])/np.linalg.norm(np.array(obs["board"]["ore"]))
                map[:, :, 2] = np.array(
                    obs["board"]["ice"])/np.linalg.norm(np.array(obs["board"]["ice"]))

                window = np.ones((11, 11, 3))
                for i in range(0, 5):
                    window[i+1:10-i, i+1:10-i, 2] = 2*i * \
                        np.ones((9-2*i, 9-2*i))
                    window[i+1:10-i, i+1:10-i, 1] = i * \
                        np.ones((9-2*i, 9-2*i))
                    window[i+1:10-i, i+1:10-i, 0] = -i*np.ones((9-2*i, 9-2*i))
                window[5:8, 5:8, 1:] = np.zeros((3, 3, 2))

                final = np.zeros((48, 48, 3))
                for i in range(3):
                    final[:, :, i] = scipy.ndimage.convolve(
                        map[:, :, i], window[:, :, i], mode='constant')
                final = np.sum(final, axis=2)
                final = final*obs["board"]["valid_spawns_mask"]

                spawn_loc = np.where(final == np.amax(final))
                spawn_loc = np.array(
                    [spawn_loc[0][0].item(), spawn_loc[1][0].item()])
                if (spawn_loc not in potential_spawns):
                    spawn_loc = potential_spawns[np.random.randint(
                        0, len(potential_spawns))]

                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, state, remainingOverageTime: int = 60):
        features = state[0][self.player]
        obs = state[1]

        image_features = features["image_features"].to(self.device)
        global_features = features["global_features"].to(self.device)

        units = obs[self.player]["units"][self.player].values()
        factories = obs[self.player]["factories"][self.player].values()

        unit_output, factory_output = self.PPO.select_action(image_features, obs)

        action = outputs_to_actions(unit_output.detach().cpu(), factory_output.detach().cpu(), units, factories, obs, factory_map = image_features[1].detach().cpu().numpy()) #Second channel is always factory_map
        return action
