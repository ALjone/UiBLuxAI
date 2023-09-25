from lux.kit import obs_to_game_state
from lux.utils import my_turn_to_place_factory
import gym
import numpy as np
from scipy.spatial.distance import cdist
from actions.actions import UNIT_ACTION_IDXS
import scipy

class SinglePlayerEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, config) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)

        self.config = config

    #NOTE: Thanks to ChatGPT
    def get_closest(self, ice, ore, valid_spawns):
        ice = np.argwhere(ice == 1)
        ore = np.argwhere(ore == 1)

        ice_dist = cdist(valid_spawns, ice, metric='cityblock').min(1)
        ore_dist = cdist(valid_spawns, ore, metric='cityblock').min(1)

        dist = np.sqrt(ice_dist**2+ore_dist**2)
        return valid_spawns[np.argmin(dist)] 

    def get_factory(self, step, player):
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

                ice = obs["board"]["ore"]
                ore = obs["board"]["ice"]

                best_coord = self.get_closest(ice, ore, potential_spawns)



                return dict(spawn=best_coord, metal=150, water=150)
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

        if self.env.state.real_env_steps >= self.config["max_game_length"]:
            done = {"player_0": True, "player_1": True}
        
        info = {}
        if done[agent]:
            info["stats"] = self.env.state.stats[agent]
            info["episode_length"] = self.env.state.real_env_steps

        return obs, reward, done[agent], info

    def reset(self, **kwargs):
        #TODO: Add exploring starts
        #Using queues for each 100th timestep (100, 200, 300 etc.) #But maybe also skew it towards later stages
        #Saving obs in these and loading from it 
        #Decide in config how often to load from these

        obs = self.env.reset(**kwargs)
        step = 0
        while self.env.state.real_env_steps < 0:
            a = {"player_0" : self.get_factory(step, "player_0"), "player_1" : self.get_factory(step, "player_1")}
            obs, _, _, _ = self.step(a)
            step += 1
        
        self.env.state.stats["player_0"]["max_value_observation"] = 0

        self.env.state.stats["player_0"]["actions"] = {
                                                        "factories": [0]*(3), #Minus one because the do nothing action is never registered here
                                                        "units" : [0]*UNIT_ACTION_IDXS,
                                                        "average_power_when_recharge": []
                                                    }
        return obs