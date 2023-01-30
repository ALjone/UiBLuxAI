import imageio

def animate(imgs):
    video_name = 'agents.gif'
    for _ in range(10):
        imgs.append(imgs[-1])

    imageio.mimsave(video_name, imgs, fps=100)