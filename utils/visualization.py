import imageio
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from wrappers.observation_wrapper import DIM_NAMES
from actions.actions import MOVES

from PIL import Image, ImageDraw, ImageFont

def annotate_image(image, frame_number):
    # Convert the image to a PIL Image
    image_pil = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image_pil)
    
    # Choose some parameters to use for the annotation
    font_size = 20
    font = ImageFont.load_default()
    text = f"Frame {frame_number}"
    text_position = (image.shape[1] - 80, 10)
    
    # Annotate the image
    draw.text(text_position, text, font=font, fill=(255, 255, 255))
    
    # Convert the PIL Image back to a numpy array
    annotated_image = np.array(image_pil)
    
    return annotated_image

def animate(imgs, video_name="agents.gif"):
    annotated_imgs = []
    for i, img in enumerate(imgs):
        annotated_img = annotate_image(img, i)
        annotated_imgs.append(annotated_img)
    
    for _ in range(10):
        annotated_imgs.append(annotated_imgs[-1])
    
    imageio.mimsave(video_name, annotated_imgs, fps=100, loop = 0)

def visualize_obs(state, step, save_img = False):
    image = state["features"].squeeze()
    size = len(DIM_NAMES)
    axs_size = ceil(np.sqrt(size))
    fig, axs = plt.subplots(axs_size, axs_size, figsize = (25, 25))
    for i, name in enumerate(DIM_NAMES):
        row, col = divmod(i, axs_size)
        axs[row, col].set_title(name)
        img = axs[row, col].imshow(image[i].T)
        plt.colorbar(img)
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


def visualize_network_output(unit_action, factory_action, unit_mask, factory_mask, step):

    fig, axs = plt.subplots(5, 4, figsize = (25, 25))
    for i, name in enumerate(MOVES):
        row, col = divmod(i, 4)
        axs[row, col].set_title(name)
        img = axs[row, col].imshow(unit_action[:, :, i].T)
        plt.colorbar(img)

    for i, name in enumerate(["LIGHT", "HEAVY", "LICHEN", "NOTHING"]):
        axs[3, i].set_title(name)
        img = axs[3, i].imshow(factory_action[:, :, i].T)
        plt.colorbar(img)
    
    axs[4, 1].set_title("Unit mask")
    axs[4, 1].imshow(unit_mask.squeeze().T)
    axs[4, 2].set_title("Factory mask")
    axs[4, 2].imshow(factory_mask.squeeze().T)

    fig.suptitle(f"Step: {step}")

    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.cla()
    plt.close()
    return data


def visualize_critic_and_returns(returns, critic_output, rewards, single_plot=True):
    if single_plot:
        plt.title("Returns, Critic and Difference")
        plt.plot(returns, label='Returns')
        plt.plot(critic_output, label='Critic Output')
        plt.plot(np.abs(np.array(returns).squeeze() - np.array(critic_output).squeeze()), label='Difference')
        plt.legend()
        plt.show()
    else:
        fig, axs = plt.subplots(3, 1, sharey=True)
        axs[0].set_title("Returns")
        axs[1].set_title("Critic guess")
        axs[2].set_title("Difference")

        axs[0].plot(returns)
        axs[1].plot(critic_output)
        axs[2].plot(np.abs(np.array(returns).squeeze() - np.array(critic_output).squeeze()))

        plt.show()

    plt.title("Rewards")
    plt.plot(rewards, label = "Rewards")
    plt.legend()
    plt.show()


def visualize_stats(lichen_player, lichen_opponent, water):
    plt.title("Stats")
    
    #Lichen
    lichen_player = np.array(lichen_player)
    lichen_opponent = np.array(lichen_opponent)
    lichen_player_max = np.max(lichen_player)
    lichen_opponent_max = np.max(lichen_opponent)
    lichen_max = max(lichen_player_max, lichen_opponent_max)
    lichen_player /= lichen_max
    lichen_opponent /= lichen_max

    plt.plot(lichen_player, label = f"Lichen player (Max {round(lichen_player_max.item())})")
    plt.plot(lichen_opponent, label = f"Lichen Opponent(Max {round(lichen_opponent_max.item())})")
    water = np.array(water)
    water_max = np.max(water)
    water /= water_max
    plt.plot(water, label = f"Water (Max {round(water_max.item())})")
    plt.legend()

    plt.show()