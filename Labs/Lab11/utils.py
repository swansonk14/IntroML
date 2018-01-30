from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def render_agent(env, agent, gif_path=None):
    """Render an animation of an agent acting in an environment and save a GIF of it.

    Arguments:
        env(object): A gym game environment.
        agent(object): An agent which acts in a gym game environment.
        gif_path(str): The path where a GIF of the agent acting in the
                       environment will be saved.
    """

    state = env.reset()
    frames = []

    while True:
        frames.append(env.render(mode='rgb_array'))
        action = agent.act(state)
        state, _, game_over, _ = env.step(action)

        if game_over:
            break

    if gif_path is not None:
        save_gif(gif_path, frames)

def save_gif(gif_path, frames):
    """Save a GIF of a sequence of image frames.

    Arguments:
        gif_path(str): The path where the GIF will be saved.
        frames(list): A list of image frames (numpy matrices).
    """

    print('Saving GIF')

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(gif_path, fps=70, writer='imagemagick')
