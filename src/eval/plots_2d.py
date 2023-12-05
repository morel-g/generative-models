import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from .plot_utils import (
    save_scatter,
    save_scatter_motion,
    make_meshgrid,
    save_velocity_fields,
    save_discrete_motion,
    save_discrete_density,
)
from src.case import Case
from src.utils import ensure_directory_exists
from src.data_manager.data_type import (
    toy_discrete_data_type,
    toy_continuous_data_type,
)
from src.eval.plot_utils import get_titles

# import ot
import pytorch_lightning as pl


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_2d(
    net: pl.LightningModule,
    output_dir: str,
    name: str,
    nb_samples: Optional[int] = 10000,
) -> None:
    """
    Sample 2D data using a given network and save a plot of the samples.

    Parameters:
    - net: The network used for sampling.
    - output_dir: The directory where the plot will be saved.
    - name: The name of the plot.
    - nb_samples: The number of samples to generate (default is 10000).
    """

    ensure_directory_exists(output_dir)

    x = net.sample(nb_samples)
    data_type = net.params.data_type

    if data_type in toy_continuous_data_type:
        # Check if net is augmented and unpack values accordingly
        x, v = (x, None) if not net.is_augmented() else x
        save_scatter(
            x.cpu().numpy(),
            output_dir,
            color="blue",
            name=name,
            s=3.0,
        )
    elif data_type in toy_discrete_data_type:
        nb_tokens = net.params.model_params["nb_tokens"]
        save_discrete_density(
            x.cpu().numpy(),
            nb_tokens,
            output_dir,
            name=name,
        )
    else:
        raise ValueError(f"Unkown data type: {data_type}")


# -----------------------------------------------------------------------------
# Discrete outputs
# -----------------------------------------------------------------------------


def compute_discrete_outputs_2d(
    net: pl.LightningModule, X: torch.Tensor, output_dir: str
) -> None:
    """
    Compute discrete 2D outputs and save them using the given network
    and input data.

    Parameters:
    - net: The network used for computation.
    - X: The input data.
    - output_dir: The directory where the outputs will be saved.
    """
    ensure_directory_exists(output_dir)
    original_font_size = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 18})

    nb_tokens = net.params.model_params["nb_tokens"]

    sampled_traj = net.sample(nb_samples=10000, return_trajectories=True)
    titles = get_titles(net, forward=False)
    save_discrete_density(
        sampled_traj[-1].cpu(),
        nb_tokens,
        output_dir,
        name="Final_density.png",
    )
    save_discrete_motion(
        sampled_traj, output_dir, nb_tokens, name="Trajectories", titles=titles
    )

    sampled_traj = net.sample(nb_samples=10000)

    x_forward_exact = net.forward_pass(
        X[:5000], use_training_velocity=False, return_trajectories=True
    )
    titles = get_titles(net, forward=True)
    save_discrete_motion(
        x_forward_exact,
        output_dir,
        nb_tokens,
        name="Exact forward motion",
        titles=titles,
    )

    save_discrete_density(
        X[:10000].cpu(),
        nb_tokens,
        output_dir,
        name="True_density.png",
    )

    # Restore original font size
    plt.rcParams.update({"font.size": original_font_size})


# -----------------------------------------------------------------------------
# Continuous outputs
# -----------------------------------------------------------------------------


def compute_continuous_outputs_2d(
    net: pl.LightningModule, X: torch.Tensor, output_dir: str
) -> None:
    """
    Compute continuous 2D outputs and save them using the given network
    and input data.

    Parameters:
    - net: The network used for computation.
    - X: The input data.
    - output_dir: The directory where the outputs will be saved.
    """
    ensure_directory_exists(output_dir)
    # Initialize
    params = net.params
    bound, s = get_bounds_and_s(params.data_type)
    bounds = (-bound, bound, -bound, bound)

    # Save velocities for non-augmented networks
    if not net.is_augmented():
        save_velocities_2d(net, output_dir)

    nb_samples = 10000
    X = X.to(net.device)

    if net.params.model_type in (
        Case.score_model,
        Case.score_model_critical_damped,
    ):
        compute_velocity_outputs(net, X, output_dir, bounds, s, nb_samples=nb_samples)
    else:
        compute_general_outputs(net, X, output_dir, bounds, s, nb_samples=nb_samples)

    # Save scatter plots
    save_scatter(
        X[:nb_samples].cpu(),
        output_dir,
        color="blue",
        name="True_density.png",
        extent=bounds,
        s=s,
    )

    x_forward_exact = net.forward_pass(
        X[:5000], use_training_velocity=False, return_trajectories=True
    )
    save_trajectories(
        net,
        x_forward_exact,
        output_dir,
        bounds,
        s,
        name="Exact forward",
        forward=True,
    )


def compute_general_outputs(
    net: pl.LightningModule,
    X: torch.Tensor,
    output_dir: str,
    bounds: tuple,
    s: float,
    nb_samples: int = 10000,
) -> None:
    """
    Compute general outputs and save trajectory information.

    Parameters:
    - net: The network used for computation.
    - X: The input data.
    - output_dir: The directory where the output will be saved.
    - bounds: Tuple representing the boundaries for plotting.
    - s: Scaling factor for plotting.
    - nb_samples: The number of samples to consider (default is 10000).
    """

    ensure_directory_exists(output_dir)

    # Save original font size and update it only for this plot
    original_font_size = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 11})

    # Save trajectories information
    save_trajectories_infos(
        net, X, output_dir, bounds, s, name="", nb_samples=nb_samples
    )

    # Restore original font size
    plt.rcParams.update({"font.size": original_font_size})


# -----------------------------------------------------------------------------
# Trajectory Functions
# -----------------------------------------------------------------------------


def save_trajectories(
    net: pl.LightningModule,
    x_traj: torch.Tensor,
    output_dir: str,
    bounds: tuple,
    s: float,
    name: str,
    forward: bool = False,
) -> None:
    """
    Save trajectory and velocity information for visualization.

    Parameters:
    - net: The network model used for computation.
    - x_traj: Trajectory data.
    - output_dir: Directory where the output will be saved.
    - bounds: Bounds for the plot.
    - s: Marker size for scatter plot.
    - name: Suffix for the file name.
    - forward: Whether the trajectory is forward or not (default is False).
    """

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Separate trajectory and velocity if network is augmented
    x_traj, v_traj = (x_traj, None) if not net.is_augmented() else x_traj

    # Get time values
    titles = get_titles(net, forward)

    # Save plots
    save_scatter_motion(
        x_traj,
        output_dir,
        color="blue",
        name="Trajectories" + name,
        extent=bounds,
        s=s,
        titles=titles,
    )

    if v_traj is not None:
        save_scatter_motion(
            v_traj,
            output_dir,
            color="blue",
            name="Velocity trajectories" + name,
            extent=bounds,
            s=s,
            titles=titles,
        )


def save_trajectories_infos(
    net: pl.LightningModule,
    X: torch.Tensor,
    output_dir: str,
    bounds: tuple,
    s: float,
    name: Optional[str] = "",
    nb_samples: int = 10000,
) -> None:
    """
    Save information about trajectories, including plots and costs.

    Parameters:
    - net: The neural network model.
    - X: Input data.
    - output_dir: Output directory.
    - bounds: Plotting boundaries.
    - s: Scatter plot marker size.
    - name: Optional name suffix for saving files.
    - nb_samples: Number of samples to generate.
    """

    # Ensure the output directory exists
    ensure_directory_exists(output_dir)

    # Prepare the name suffix
    name_suffix = f"_{name}" if name else ""

    # Sample trajectories from the model
    sampled_traj = net.sample(nb_samples, return_trajectories=True)

    # Convert to NumPy arrays and separate x and v if augmented
    if net.is_augmented():
        x_traj, v_traj = map(lambda x: x.cpu().numpy(), sampled_traj)
    else:
        x_traj = sampled_traj.cpu().numpy()
        v_traj = None

    # Save trajectory and other plots
    save_trajectories(
        net,
        (x_traj, v_traj) if v_traj is not None else x_traj,
        output_dir,
        bounds,
        s,
        name_suffix,
    )
    final_x_traj = x_traj[-1]
    save_scatter(
        final_x_traj,
        output_dir,
        color="blue",
        name="Samples" + name_suffix,
        extent=bounds,
        s=s,
    )


# -----------------------------------------------------------------------------
# Velocity Functions
# -----------------------------------------------------------------------------


def compute_velocity_outputs(
    net: pl.LightningModule,
    X: torch.Tensor,
    output_dir: str,
    bounds: tuple,
    s: float,
    nb_samples: int = 10000,
) -> None:
    """
    Compute velocity outputs and save trajectory information based on different backward schemes.

    Parameters:
    - net: The network used for computation.
    - X: The input data.
    - output_dir: The directory where the output will be saved.
    - bounds: Tuple representing the boundaries for plotting.
    - s: Scaling factor for plotting.
    - nb_samples: The number of samples to consider (default is 10000).
    """

    ensure_directory_exists(output_dir)

    default_backward_scheme = net.get_backward_scheme()

    backward_schemes = net.get_backward_schemes()
    for scheme in backward_schemes:
        net.set_backward_scheme(scheme)
        save_trajectories_infos(
            net,
            X,
            output_dir,
            bounds,
            s,
            name=scheme,
            nb_samples=nb_samples,
        )

    net.set_backward_scheme(default_backward_scheme)


def save_velocities_2d(
    net: pl.LightningModule,
    output_dir: str,
    bound: float = 3.5,
    nb_points: int = 30,
) -> None:
    """Save 2D velocities from a network model.

    Args:
        net: The network object to get velocities from.
        output_dir: The directory where to save the outputs.
        bound: The boundary for making the mesh grid.
        nb_points: Number of points for each dimension in the mesh grid.
    """
    # Ensure the output directory exists
    ensure_directory_exists(output_dir)

    # Create mesh grid and move to appropriate device
    mesh, x_axis, y_axis = make_meshgrid(bound, nb_points)
    mesh_tensor = torch.tensor(mesh).float().to(net.device)

    # Get velocity scores and neural net outputs
    velocity_scores = net.get_velocities(mesh_tensor).cpu().numpy()
    neural_net_output = net.get_neural_net(mesh_tensor).cpu().numpy()

    # Retrieve and format trajectory times for titles
    titles = get_titles(net, forward=False)

    # Save velocity fields for scores
    save_velocity_fields(
        velocity_scores,
        [x_axis, y_axis],
        output_dir,
        normalize=True,
        titles=titles,
        name="Scores.gif",
    )

    # Save velocity fields for neural net output
    save_velocity_fields(
        neural_net_output,
        [x_axis, y_axis],
        output_dir,
        normalize=True,
        titles=titles,
        name="Neural_net.gif",
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_bounds_and_s(data_type: str) -> tuple:
    """Return the appropriate bounds and scaling factor for a given data type."""
    return (4.5, None) if data_type != Case.multimodal_swissroll else (1.0, 3.0)
