from lux.kit import obs_to_game_state
from lux.utils import my_turn_to_place_factory
import gym
import numpy as np
from actions.idx_to_lux_move import UNIT_ACTION_IDXS, FACTORY_ACTION_IDXS


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
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def step(self, action):
        agent = self.agents[0]
        opp_agent = self.agents[1]

        opp_factories = self.env.state.factories[opp_agent]

        for k in opp_factories:
            factory = opp_factories[k]
            factory.cargo.water = 1000 # set enemy factories to have 1000 water to keep them alive the whole around and treat the game as single-agent

        if self.state.real_env_steps < 0:
            opp_action = self.get_factory(self.state.env_steps, self.env.state.env_cfg, self.state.get_obs(), opp_agent)
        else:
            opp_action = {}

        action = {agent: action, opp_agent: opp_action}

        obs, reward, done, info = super().step(action)

        #Stats

        self.prev_actions = action
        units = self.env.state.units[agent]
        for unit_id, act in action[agent].items():
            if "unit" in unit_id:
                act = act[0] #Because of action queue
                self.env.state.stats[agent]["actions"]["units"][act[0]] += 1
                #Recharge action if the first element in the action array is 5
                if act[0] == 5 and unit_id in units.keys(): #No idea why this check is needed??? Maybe it died
                    self.env.state.stats[agent]["actions"]["average_power_when_recharge"].append(units[unit_id].power)
            elif "factory" in unit_id:
                self.env.state.stats[agent]["actions"]["factories"][act] += 1

        return obs, reward, done[agent], info[agent]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_actions = {}
        self.env.state.stats["player_0"]["actions"] = {
                                                        "factories": [0]*(FACTORY_ACTION_IDXS-1), #Minus one because the do nothing action is never registered here
                                                        "units" : [0]*UNIT_ACTION_IDXS,
                                                        "average_power_when_recharge": []
                                                    }

        return (obs[0][self.agents[0]], obs[1])