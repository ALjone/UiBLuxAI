from lux.kit import obs_to_game_state
from lux.utils import my_turn_to_place_factory
import gym
import numpy as np
from actions.actions import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS
import scipy

class SinglePlayerEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)

    def get_factory(self, step, env_cfg, obs, player):
        if step == 0:
                # bid 0 to not waste resources bidding and declare as the default faction
                return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, env_cfg, obs)
            # factory placement period

            # how many factories you have left to place
            factories_to_place = game_state.teams[player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                return dict(spawn=spawn_loc, metal=150, water=150) #Metal and water being 1000 ensures that our opponent only has 1 factory
            return dict()

    def step(self, action):
        agent = self.agents[0]
        opp_agent = self.agents[1]

        opp_factories = self.env.state.factories[opp_agent]

        for k in opp_factories:
            factory = opp_factories[k]
            factory.cargo.water = 1000 # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent

        obs, reward, done, info = self.env.step(action)
        self.prev_actions = action
        
        info = {}
        if done[agent]:
            info["stats"] = self.env.state.stats[agent]
            info["episode_length"] = self.env.state.real_env_steps


        return obs, reward, done[agent], info
    

    def early_setup(self, step: int, player):
        obs = self.state.get_obs()
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period


            # how many factories you have left to place
            factories_to_place = game_state.teams[player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                
                #return dict(spawn = potential_spawns[0], metal = 150, water = 150)

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

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        step = 0
        while self.env.state.real_env_steps < 0:
            a = self.early_setup(step, "player_0")
            step += 1
            a = {"player_0" : a, "player_1" : self.get_factory(self.state.env_steps, self.env.state.env_cfg, self.state.get_obs(), "player_1")}
            obs, _, _, _ = self.step(a)

        self.prev_actions = {}
        self.env.state.stats["player_0"]["actions"] = {
                                                        "factories": [0]*(FACTORY_ACTION_IDXS-1), #Minus one because the do nothing action is never registered here
                                                        "units" : [0]*UNIT_ACTION_IDXS,
                                                        "average_power_when_recharge": []
                                                    }
        return obs