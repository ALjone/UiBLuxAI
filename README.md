# UiBLuxAI

https://docs.google.com/presentation/d/1y-0tbCL7tE0Zk4FrThTu_77o_NypuFvNtzr4wUXGyIc/edit?usp=sharing

## State

### Image features

Every feature here has a 48x48 channel containing the information in every tile.

1. `int` Friendly / Enemy unit mask  -  1 for friendly, -1 for enemy, 0 else
2. `int` Friendly / Enemy factory mask  -  1 for friendly, -1 for enemy, 0 else
3. `float` Unit ice cargo -  % of cargo filled up with ice
4. `float` Unit ore cargo -  % of cargo filled up with ore
5. `float` Unit power cargo -  % of cargo filled up with power
6. `float` Unit water cargo -  % of cargo filled up with water
7. `float` Unit metal cargo -  % of cargo filled up with metal
8. `float` Factory ice cargo -  % of cargo filled up with ice
9. `float` Factory ore cargo -  % of cargo filled up with ore
10. `float` Factory power cargo -  % of cargo filled up with power
11. `float` Factory water cargo -  % of cargo filled up with water
12. `float` Factory metal cargo -  % of cargo filled up with metal
13. `int` Tile ice content -  amount of ice on each tile
14. `int` Tile ore content  -  amount of ore on each tile
15. `int` Tile water content  -  amount of water on each tile
16. `int` Tile metal content  -  amonut of metal on each tile
17. `int` Tile friendly lichen content  -  amount of friendly lichen on each tile
18. `int` Tile enemy lichen content  -  amount of enemy lichen on each tile
19. `int` Friendly unit light  -  1 for light unit, 0 else
20. `int` Friendly unit heavy  -  1 for heavy unit, 0 else
21. `int` Enemy unit light  -  1 for light unit, 0 else
22. `int` Enemy unit heavy  -  1 for heavy unit, 0 else
23. `int` Action queue  -  Friendly units that have an action queue
24. `int` Lenght of queue  -  Lenght of action queue for each friendly unit on map
25. `int` Friendly next step  -  Position of every unit with an action queue in the next timestep


### Global features
1. `float` Day -  This is a cyclic feature. See this method: https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
2. `float` Night -  Part of number first feature
3. `float` Timestep  -  % of total timesteps we are currently on
4. `float` Lichen distribution  -  Distribution of lichen for each team. 1 means agent has all lichen, -1 means enemy agent has all lichen. 0 means equal lichen. Can be computed by 2 * friendly/total - 1
5. `int` Day/Night  -  Day is represented by 1, night by 0
6. `int` #Friendly factories  -  Number of friendly factories in total on the map at this timestep
7. `int` #Enemy factories  -  Number of enemy factories in total on the map at this timestep
8. `int` #Friendly units  -  Number of friendly units in total on the map at this timestep
9. `int` #Enemy units  -  Number of enemy units in total on the map at this timestep
10. `int` #Ice on map  -  Amount of ice in total on the map
11. `int` #Ore on map  -  Amount of ore in total on the map
12. `int` #Rubble on map  -  Amount of rubble in total on the map
13. `float` Entropy of Ice  -  Statistical entropy of the ice on the map
14. `float` Entropy of Ore  -  Statistical entropy of the ore on the map
13. `float` Entropy of Rubble  -  Statistical entropy of the rubble on the map


## Action Space

1. Infinite move-loop. 
>>>One for each direction [NORTH, SOUTH, EAST, WEST]
2. Pickup.
>>>One for each (resource, amount) combination. Resources = [POWER], Amounts = [50%, 100%]
3. Self Destruct
4. No Action.
>>>Does nothing, allows previous action queues to execute.
5. Walk to closest ice.
>>> Computes shortest path to ice (shortest w.r.t power or time???) and creates a sequence of directions towards it (better with long walks along one dim at the time due to limited queue length and option for looping)
6. Digg untill full or resource is empty.
>>> Compute minimum of the two and digg untill this limit is reached.
7. Deliver resources to closest factory.
>>> Compute path to closest factory (shortest w.r.t time for now) and create sequence of directions to this point, finish with transfer of all resources.

