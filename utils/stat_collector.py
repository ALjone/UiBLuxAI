import numpy as np

class StatCollector:
    def __init__(self, player) -> None:
        
        self.player = player
        #Consumption
        self.power_consumed = []

        #Generation
        self.ice_mined = []
        self.ore_mined = []
        self.lichen_made = []
        self.light_robots_built = []
        self.heavy_robots_built = []
        self.metal_made = []
        self.water_made = []

        #Destroyed
        self.rubble_destroyed = []
        self.lichen_destroyed = []
        self.light_robots_lost = []
        self.heavy_robots_lost = []
        self.factories_lost = []

        #Picked up
        self.power_picked_up = []
        self.other_picked_up = [] #Anything not power, i.e ore, ice, water, metal

        #Transfer
        self.ice_transfered = []
        self.ore_transfered = []
        self.power_transfered = []

        #Rewards
        self.factory_punishment = []
        self.unit_reward = []
        self.resources_reward = []
        self.rubble_reward = []
        self.win_reward = []
        self.env_reward = []

        #Other
        self.max_value_observation = []

    def update(self, stats):
        """Call this with the stats dict at the end of an episode"""
        #If we're already giving the stats for player0, we skip this part
        if self.player in stats.keys():
            stats = stats[self.player]

        #Consumption
        self.power_consumed.append(sum([val for val in stats["consumption"]["power"].values()])) #Loop over LIGHT, HEAVY, FACTORY

        #Generation
        gen = stats["generation"]
        self.ice_mined.append(sum([val for val in gen["ice"].values()])) #Sum over LIGHT, HEAVY 
        self.ore_mined.append(sum([val for val in gen["ore"].values()])) #Sum over LIGHT, HEAVY
        self.lichen_made.append(gen["lichen"])
        self.light_robots_built.append(gen["built"]["LIGHT"])
        self.heavy_robots_built.append(gen["built"]["HEAVY"])
        self.metal_made.append(gen["metal"])
        self.water_made.append(gen["water"])

        #Destroyed
        dest = stats["destroyed"]
        self.rubble_destroyed.append(sum([val for val in dest["rubble"].values()]))
        self.lichen_destroyed.append(sum([val for val in dest["lichen"].values()]))
        self.light_robots_lost.append(dest["LIGHT"])
        self.heavy_robots_lost.append(dest["HEAVY"])
        self.factories_lost.append(dest["FACTORY"])

        #Picked up
        total = sum([val for val in stats["pickup"].values()])
        power = stats["pickup"]["power"]
        self.power_picked_up.append(power)
        self.other_picked_up.append(total-power)

        #Transfer
        transf = stats["transfer"]
        self.ice_transfered.append(transf["ice"])
        self.ore_transfered.append(transf["ore"])
        self.power_transfered.append(transf["power"])

        #Rewards
        reward = stats['rewards']
        self.factory_punishment.append(reward["factory_punishment"])
        self.unit_reward.append(reward["unit_reward"])
        self.resources_reward.append(reward['resource_reward'])
        self.rubble_reward.append(reward['rubble_reward'])
        self.win_reward.append(reward["win_reward"])
        self.env_reward.append(reward["env_reward"])

        #Other
        self.max_value_observation.append(stats["max_value_observation"])


    def get_last_x(self, x):
        return {"main":
                    {
                    "metal_made": self.metal_made[-x:],
                    "water_made": self.water_made[-x:],
                    "lights_surviving": [np.mean(self.light_robots_built[-x:]) - np.mean(self.light_robots_lost[-x:])], #List because we take mean of it later
                    "heavies_surviving": [np.mean(self.heavy_robots_built[-x:]) - np.mean(self.heavy_robots_lost[-x:])] #List because we take mean of it later
                    },
            
                "consumption" : 
                    {
                    "power_consumed": self.power_consumed[-x:]
                    },
                
                "generation": 
                    {
                    "ice_mined": self.ice_mined[-x:],
                    "ore_mined": self.ore_mined[-x:],
                    "lichen_made": self.lichen_made[-x:],
                    "light_robots_built": self.light_robots_built[-x:],
                    "heavy_robots_built": self.heavy_robots_built[-x:],
                    "metal_made": self.metal_made[-x:],
                    "water_made": self.water_made[-x:]
                    },

                "destroyed": 
                    {
                    "rubble_destroyed": self.rubble_destroyed[-x:],
                    "lichen_destroyed": self.lichen_destroyed[-x:],
                    "light_robots_lost": self.light_robots_lost[-x:],
                    "heavy_robots_lost": self.heavy_robots_lost[-x:],
                    "factories_lost": self.factories_lost[-x:]
                    },

                "pickup": 
                    {
                    "power_picked_up": self.power_picked_up[-x:],
                    "other_picked_up": self.other_picked_up[-x:]
                    },

                "transfered": 
                    {
                    "ice_transfered": self.ice_transfered[-x:],
                    "ore_transfered": self.ore_transfered[-x:],
                    "power_transfered": self.power_transfered[-x:]
                    },

                "rewards":
                    {
                    "factory_punishment": self.factory_punishment[-x:],
                    "unit_reward": self.unit_reward[-x:],
                    "resources_reward": self.resources_reward[-x:],
                    "rubble_reward": self.rubble_reward[-x:],
                    "win_reward": self.win_reward[-x:],
                    "env_reward": self.env_reward[-x]
                    },
                
                "other":
                    {
                    "max_value_observation": self.max_value_observation[-x:],
                    }
                }
    
    def __repr__(self) -> str:
        string = "Average over last 100 episodes:\n"
        categories = self.get_last_x(100)
        for category_name, category in categories.items():
            string += f"\n{category_name}:"
            for name, value in category.items():
                string += f"\t{name}: {round(np.mean(value).item(), 3)}\n"
