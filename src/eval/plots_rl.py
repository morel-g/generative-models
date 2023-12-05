import torch
import pytorch_lightning as pl
import os
import gym
from torch.utils.data import DataLoader

from src.utils import ensure_directory_exists
from src.eval.rl_eval_utils.eval_hoppper import MuJoCoRenderer, save_samples_with_render
from src.eval.rl_eval_utils.eval_maze2d import Maze2dRenderer
from src.eval.plot_utils import get_titles
from src.data_manager.rl_data_utils import RLDataUtils

FIG_DIR = "figures/"


@torch.no_grad()
def compute_rl_outputs(
    net: pl.LightningModule,
    val_dataset: DataLoader,
    output_dir: str,
):
    """
    Computes and saves text outputs generated by the neural network.

    Parameters:
    - net (pl.LightningModule): The neural network model.
    - val_dataset (DataLoader): The validation dataset.
    - output_dir (str): The directory to save the output.

    Returns:
    - None
    """
    fig_dir = os.path.join(output_dir, FIG_DIR)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    for x in val_dataloader:
        x = RLDataUtils.denormalize(x)
        save_rl_samples(
            net.params.data_type,
            x.cpu().numpy(),
            fig_dir,
            name="True_samples.gif",
        )
        break

    save_rl_traj(net, fig_dir, nb_samples=3)


def save_rl_traj(
    net: pl.LightningModule, output_dir: str, name: str = "Hopper", nb_samples: int = 5
):
    net.set_trajectory_length(4)
    x_traj = net.sample(nb_samples=nb_samples, return_trajectories=True).cpu().numpy()
    save_rl_samples(
        net.params.data_type,
        x_traj[-1],
        output_dir,
        name=name + "_samples.gif",
    )
    # Transpose the trajectories to have batch dim first
    new_dims = (1, 0) + tuple(range(2, x_traj.ndim))
    x_traj = x_traj.transpose(*new_dims)
    titles = get_titles(net, forward=False)
    for i, xi in enumerate(x_traj):
        save_rl_samples(
            net.params.data_type,
            xi,
            output_dir,
            name=name + f"_traj_{i}.gif",
            titles=titles,
        )


def sample_rl(net: pl.LightningModule, output_dir: str, name: str, nb_samples: int = 5):
    x = net.sample(nb_samples).cpu().numpy()
    save_rl_samples(
        net.params.data_type,
        x,
        output_dir=output_dir,
        name=name,
    )


def save_rl_samples(
    env_name,
    observations,
    output_dir,
    name="rl_samples.gif",
    titles=None,
    speed_factor=3,
):
    ensure_directory_exists(output_dir)
    env = gym.make(env_name)
    if "maze2d" in env_name:
        render = Maze2dRenderer(env_name)
    else:
        render = MuJoCoRenderer(env)
    save_samples_with_render(
        render, observations, output_dir, name, titles, speed_factor
    )