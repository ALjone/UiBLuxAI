import imageio
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from wrappers.observation_wrapper import DIM_NAMES

def animate(imgs):
    video_name = 'agents.gif'
    for _ in range(10):
        imgs.append(imgs[-1])

    imageio.mimsave(video_name, imgs, fps=100)

def visualize_obs(obs):
    size = obs.shape[0]
    assert size == len(DIM_NAMES), f"Number of dimension names should match size of the observation"
    axs_size = ceil(np.sqrt(size))
    fig, axs = plt.subplots(axs_size, axs_size)
    for i, name in enumerate(DIM_NAMES):
        row, col = divmod(i, axs_size)
        axs[row, col].set_title(name)
        axs[row, col].plot(obs[i])
    
    plt.show()