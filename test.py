import numpy as np
from scipy.spatial.distance import cdist
def get_closest(ice, ore, valid_spawns):
    best_dist = np.inf
    best_coord = None

    ice = np.argwhere(ice == 1)
    ore = np.argwhere(ore == 1)

    ice_dist = cdist(valid_spawns, ice, metric='cityblock').min(1)
    ore_dist = cdist(valid_spawns, ore, metric='cityblock').min(1)

    dist = np.sqrt(ice_dist**2+ore_dist**2)
    return valid_spawns[np.argmin(dist)] 




    #TODO: Add penalty for distance to enemy factory
    for coord in valid_spawns:
        # calculate distances to nearest element in arr1 and arr2
        dist1 = np.min(np.sqrt(np.sum((coord - ice)**2, axis=1)))
        dist2 = np.min(np.sqrt(np.sum((coord - ore)**2, axis=1)))
        
        # calculate combined distance and update best_dist and best_coord if necessary
        ice_dist = dist1 + dist2
        if ice_dist < best_dist:
            best_dist = ice_dist
            best_coord = coord

    best_dist_vec = np.min(np.sqrt(np.sum((valid_spawns[np.argmin(dist)] - ice)**2, axis=1))) + np.min(np.sqrt(np.sum((valid_spawns[np.argmin(dist)] - ore)**2, axis=1)))

    print("Vec:", best_dist_vec)
    print("Loop:", best_dist)

    return valid_spawns[np.argmin(dist)]    
    return best_coord

p = 0.01

ice = np.random.rand(48, 48) < p
ore = np.random.rand(48, 48) < p 

mask = np.random.rand(48, 48) < p

potential_spawns = np.array(
                    list(zip(*np.where(mask == 1))))

print(get_closest(ice, ore, potential_spawns))