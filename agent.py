from lux.kit import obs_to_game_state, EnvConfig
from lux.utils import my_turn_to_place_factory
import numpy as np
import torch
import scipy
from network.model import actor
from utils.utils import outputs_to_actions, UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
from utils.wrappers import ImageWithUnitsWrapper


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig, device=torch.device("cpu")) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.device = device

        self.unit_actions_per_cell = UNIT_ACTION_IDXS
        self.factory_actions_per_cell = FACTORY_ACTION_IDXS

        self.action_model = actor(23, self.unit_actions_per_cell)
        self.factory_model = actor(23, self.factory_actions_per_cell)

        self.action_model.to(self.device)
        self.factory_model.to(self.device)

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
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))

                map = np.zeros((48, 48, 3))
                map[:, :, 0] = np.array(obs["board"]["rubble"])
                map[:, :, 1] = np.array(obs["board"]["ore"])
                map[:, :, 2] = np.array(obs["board"]["ice"])

                window = np.ones((11, 11, 3))
                for i in range(0, 5):
                    window[i+1:10-i, i+1:10-i, 1:] = i * \
                        np.ones((9-2*i, 9-2*i, 2))
                    window[i+1:10-i, i+1:10-i, 0] = -i*np.ones((9-2*i, 9-2*i))

                final = np.zeros((48, 48, 3))
                for i in range(3):
                    final[:, :, i] = scipy.signal.convolve2d(
                        map[:, :, i], window[:, :, i], mode='same')
                final = np.sum(final, axis=2)
                final = final*obs["board"]["valid_spawns_mask"]

                spawn_loc = np.where(final == np.amax(final))
                spawn_loc = np.array(
                    [spawn_loc[0].item(), spawn_loc[1].item()])
                if (spawn_loc not in potential_spawns):
                    spawn_loc = potential_spawns[np.random.randint(0)]

                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):

        image = obs["image_features"].to(self.device)
        units = obs["unit_to_id"]
        factories = obs["factory_to_id"]

        unit_output = self.action_model(image)
        factory_output = self.factory_model(image)

        # NOTE How actions are formatted
        # a[0] (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)

        return outputs_to_actions(unit_output, factory_output, units, factories)
