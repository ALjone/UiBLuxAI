import imageio
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from wrappers.observation_wrapper import DIM_NAMES

def animate(imgs, video_name = "agents.gif"):
    for _ in range(10):
        imgs.append(imgs[-1])

    imageio.mimsave(video_name, imgs, fps=100)

def visualize_obs(obs, step, global_, save_img = False):
    size = obs.shape[0]
    assert size == len(DIM_NAMES), f"Number of dimension names should match size of the observation"
    axs_size = ceil(np.sqrt(size))
    fig, axs = plt.subplots(axs_size, axs_size, figsize = (25, 25))
    for i, name in enumerate(DIM_NAMES):
        row, col = divmod(i, axs_size)
        axs[row, col].set_title(name)
        img = axs[row, col].imshow(obs[i].T)
        plt.colorbar(img)

    text = ''
    for i in global_:
        text += str(round(i.item(), 3)) + ' | '
    fig.suptitle(text)
    if save_img:
        plt.savefig(f'Images/obs-{step}.png')
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.cla()
    plt.close()
    return data
    #plt.show()