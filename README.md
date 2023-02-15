# UiBLuxAI

https://docs.google.com/presentation/d/1y-0tbCL7tE0Zk4FrThTu_77o_NypuFvNtzr4wUXGyIc/edit?usp=sharing

# Installing PyTorch with CUDA 11.8

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

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


## NEW Action Space

STILL NEEDS TO BE IMPLEMENTED!

Flattend actionarray:

0. Do Nothing
1. Move north
2. Move east
3. Move south
4. Move west
5. Transfer Ice North
6. Transfer Ice East
7. Transfer Ice West
8. Transfer Ice South
9. Transfer Ore North
10. Transfer Ore East
11. Transfer Ore South
12. Transfer Ore West
13. Transfer Water North
14. Transfer Water East
15. Transfer Water South
16. Transfer Water West
17. Transfer Power North 25%
18. Transfer Power North 50%
19. Transwer Power North 75%
20. Transwer Power North 100%
21. Transfer Power East 25%
22. Transfer Power East 50%
23. Transwer Power East 75%
24. Transwer Power East 100%
25. Transwer Power South 25%
26. Transwer Power South 50%
27. Transwer Power South 75%
28. Transwer Power South 100%
29. Transwer Power West 25%
30. Transwer Power West 50%
31. Transwer Power West 75%
32. Transwer Power West 100% 
33. Pickup Power
34. Pickup Ice
35. Pickup Ore
36. Pickup Water
37. Digg
38. Self Destruct

1. Move x 5 - One for each dir
2. Transfer Resource x 12 - [North, East, South, West] * 100% * [Water, Ice, Ore]  
3. Transfer Power x 16 - [North, East, South, West] * [25%, 50%, 75%, 100%]
4. Pickup x 4 - [Power, Ice, Ore, Water]
5. Digg x 1
6. Self Destruct x 1
