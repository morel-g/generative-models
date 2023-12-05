# Code original source: https://github.com/jannerm/diffuser/blob/main/diffuser/utils/rendering.py
import os
import warnings
import numpy as np
import einops
import gym
import imageio
import matplotlib.pyplot as plt
import math
import torch


from src.eval.plot_utils import (
    save_video,
    save_images_as_grid,
    figure_to_data,
)
from src.data_manager.rl_data_utils import RLDataUtils

MAZE_BOUNDS = {
    "maze2d-umaze-v1": (0, 5, 0, 5),
    "maze2d-medium-v1": (0, 8, 0, 8),
    "maze2d-large-v1": (0, 9, 0, 12),
}

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env


# -----------------------------------------------------------------------------#
# ------------------------------ helper functions -----------------------------#
# -----------------------------------------------------------------------------#


def create_maze2d_cond(nb_samples):
    state_start = RLDataUtils.create_random_states(nb_samples, 0)
    state_end = RLDataUtils.create_random_states(nb_samples, -1)
    return torch.stack((state_start, state_end), dim=1)

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)


def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def plot2img(fig):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype="uint8").reshape((height, width, 4))


class MazeRenderer:
    def __init__(self, env):
        if type(env) is str:
            env = load_environment(env)
        self._config = env._config
        self._background = self._config != " "
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, title=None, return_fig=False):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)

        plt.imshow(
            self._background * 0.5 + 0.5,
            extent=self._extent,
            cmap=plt.cm.binary,
            vmin=0,
            vmax=1,
        )

        # Get the number of rows and columns in the background
        rows, cols = self._background.shape
        # Iterate over each element in the background array
        for (j, i), value in np.ndenumerate(self._background):
            if value == 0.0:
                # Calculate the normalized position for the rectangle
                x = i / cols
                y = j / rows

                # Adjust the size of the rectangle to be slightly smaller than the cell
                rect_width = 1 / cols * 0.97
                rect_height = 1 / rows * 0.97

                # Center the rectangle within the cell
                x += (1 / cols - rect_width) / 2
                y += (1 / rows - rect_height) / 2

                # Add the rectangle patch at the calculated position
                plt.gca().add_patch(
                    plt.Rectangle(
                        (x, y),
                        rect_width,
                        rect_height,
                        fill=False,
                        edgecolor="white",
                        lw=0.5,
                    )
                )

        # Set the x and y limits to match the extent
        plt.xlim(0, 1)
        plt.ylim(1, 0)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0, 1, path_length))

        # plt.plot(observations[:, 1], observations[:, 0], c="black", zorder=10)
        plt.scatter(observations[:, 1], observations[:, 0], c=colors, zorder=20)
        plt.axis("off")
        plt.title(title)
        # Plot the starting points end points.
        plt.scatter(
            observations[0, 1],
            observations[0, 0],
            s=100,
            facecolors="white",
            edgecolors="black",
            zorder=30,
        )
        plt.scatter(
            observations[-1, 1],
            observations[-1, 0],
            s=100,
            marker="^",
            facecolors="white",
            edgecolors="black",
            zorder=30,
        )
        fig.subplots_adjust(
            left=0.025, bottom=0.025, right=0.975, top=0.95, wspace=0, hspace=0
        )

        if not return_fig:
            img = plot2img(fig)
            return img
        else:
            return fig

    def composite(self, savepath, paths, ncol=5, **kwargs):
        """
        savepath : str
        observations : [ n_paths x horizon x 2 ]
        """
        assert (
            len(paths) % ncol == 0
        ), "Number of paths must be divisible by number of columns"

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(
            images, "(nrow ncol) H W C -> (nrow H) (ncol W) C", nrow=nrow, ncol=ncol
        )
        imageio.imsave(savepath, images)
        print(f"Saved {len(paths)} samples to: {savepath}")


class Maze2dRenderer(MazeRenderer):
    def __init__(self, env):
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]

        observations = observations + 0.5
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f"Unrecognized bounds for {self.env_name}: {bounds}")

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, **kwargs)

    def save_state_trajectories(
        self, state_traj, output_dir, name="maze_2d", titles=None
    ):
        if name.endswith(".gif"):
            name = name[:-4]
        # Set batch_dim first
        new_dims = (1, 0) + tuple(range(2, state_traj.ndim))
        state_traj = state_traj.transpose(*new_dims)

        for i, xi in enumerate(state_traj):
            self.save_states_trajectory(
                xi, output_dir, name + f"_traj_{i}", titles=titles
            )

    def save_states_trajectory(self, x_traj, output_dir, name, titles=None):
        if not name.endswith(".gif"):
            name += ".gif"
        figures = []

        for i, xi in enumerate(x_traj):
            figures.append(
                figure_to_data(self.renders(xi, title=titles[i], return_fig=True))
            )

        save_video(output_dir, figures, name)

    def save_state_samples(self, states, output_dir, name="maze2d_sample"):
        if name.endswith(".png"):
            name = name[:-4]
        imgs = []
        for i, state in enumerate(states):
            imgs.append(self.renders(state, return_fig=False))

        num_images = len(imgs)
        nb_cols = math.ceil(math.sqrt(num_images))
        nb_rows = math.ceil(num_images / nb_cols)
        save_images_as_grid(
            imgs,
            output_dir,
            nb_rows,
            nb_cols,
            name,
        )


# -----------------------------------------------------------------------------#
# ---------------------------------- rollouts ---------------------------------#
# -----------------------------------------------------------------------------#


# def set_state(env, state):
#     qpos_dim = env.sim.data.qpos.size
#     qvel_dim = env.sim.data.qvel.size
#     if not state.size == qpos_dim + qvel_dim:
#         warnings.warn(
#             f"[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, "
#             f"but got state of size {state.size}"
#         )
#         state = state[: qpos_dim + qvel_dim]

#     env.set_state(state[:qpos_dim], state[qpos_dim:])


# def rollouts_from_state(env, state, actions_l):
#     rollouts = np.stack(
#         [rollout_from_state(env, state, actions) for actions in actions_l]
#     )
#     return rollouts


# def rollout_from_state(env, state, actions):
#     qpos_dim = env.sim.data.qpos.size
#     env.set_state(state[:qpos_dim], state[qpos_dim:])
#     observations = [env._get_obs()]
#     for act in actions:
#         obs, rew, term, _ = env.step(act)
#         observations.append(obs)
#         if term:
#             break
#     for i in range(len(observations), len(actions) + 1):
#         ## if terminated early, pad with zeros
#         observations.append(np.zeros(obs.size))
#     return np.stack(observations)
