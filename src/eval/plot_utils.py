import numpy as np
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import imageio
import os
from matplotlib.axes import Axes
from typing import List, Union, Optional, Tuple
from PIL import ImageFont

from src.utils import ensure_directory_exists
from src.data_manager.toy_data_utils import ToyDiscreteDataUtils

FIG_DIR = "figures/"
FONT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../fonts/NotoSerif-VariableFont_wdth,wght.ttf",
)


def get_font(size: int = 12) -> str:
    return ImageFont.truetype(FONT_PATH, size=size)


def get_titles(net, forward):
    # Get time values
    t = net.get_traj_times()
    time_range = reversed(range(t.shape[0])) if not forward else range(t.shape[0])

    # Prepare titles for the plot
    titles = (
        None
        if hasattr(net, "remove_titles") and net.remove_titles()
        else ["T = " + str(round(t[i].item(), 3)) for i in time_range]
    )
    return titles


def figure_to_data(fig: plt.Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to a numpy array.

    Args:
        fig (plt.Figure): The figure to convert.

    Returns:
        np.ndarray: A numpy array representation of the figure.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    width, height = fig.canvas.get_width_height()
    X = buf.reshape((height, width, 3))

    return np.transpose(X, (2, 0, 1))


def save_figure(dir_path: str, fig: plt.Figure, name: str, fig_dir: str = FIG_DIR):
    """
    Save a figure to a file.

    Args:
        dir_path (str): The directory path where the figure should be saved.
        fig (plt.Figure): The figure to save.
        name (str): The name of the file.
        fig_dir (str, optional): The directory for storing figures. Defaults to FIG_DIR.
    """
    name = name if os.path.splitext(name)[-1].lower() == ".png" else name + ".png"
    ensure_directory_exists(os.path.join(dir_path, fig_dir))
    full_path = os.path.join(dir_path, fig_dir, name)

    try:
        fig.savefig(full_path)
    except Exception as e:
        print(f"An error occurred while saving the figure: {e}")


def save_video(
    dir_path: str,
    figs: List[np.ndarray],
    name: str,
    fig_dir: str = FIG_DIR,
    fps: int = None,
):
    """
    Save multiple figures as a video.

    Args:
        dir_path (str): The directory path where the video should be saved.
        figs (List[np.ndarray]): A list of numpy arrays representing the figures.
        name (str): The name of the video file.
        fig_dir (str, optional): The directory for storing figures. Defaults to FIG_DIR.
        fps (int, optional): Frames per second. Calculated based on the number of frames if not provided.
    """
    name = name if os.path.splitext(name)[-1].lower() == ".gif" else name + ".gif"
    ensure_directory_exists(os.path.join(dir_path, fig_dir))
    full_path = os.path.join(dir_path, fig_dir, name)

    fps = fps if fps else (10 if len(figs) > 30 else 7)
    duration = 1000.0 / fps

    try:
        imageio.mimsave(
            full_path,
            [np.swapaxes(np.swapaxes(f, 0, 1), 1, -1) for f in figs],
            duration=duration,
            loop=0,
        )
    except Exception as e:
        print(f"An error occurred while saving the video: {e}")


def save_images_as_grid_video(
    imgs: List[np.ndarray],
    output_dir: str,
    nb_rows: int,
    nb_cols: int,
    name: str = "",
    fig_dir: str = FIG_DIR,
):
    """
    Save images as a video arranged in a grid layout.

    Args:
        imgs (List[np.ndarray]): A list of images to include in the video.
        output_dir (str): The directory path where the video should be saved.
        nb_rows (int): The number of rows in the grid layout.
        nb_cols (int): The number of columns in the grid layout.
        name (str, optional): The name of the video file. Defaults to an empty string.
        fig_dir (str, optional): The directory for storing figures. Defaults to FIG_DIR.
    """
    figs = []
    for img in imgs:
        fig = arrange_images_in_grid(img, nb_rows, nb_cols)
        figs.append(figure_to_data(fig))
        plt.close(fig)

    save_video(output_dir, figs, name, fig_dir=fig_dir)


def arrange_images_in_grid(imgs, nb_rows: int, nb_cols: int) -> plt.Figure:
    """
    Create a figure from a list of images arranged in a grid.

    Args:
        imgs (List[np.ndarray]): List of images to display.
        nb_rows (int): Number of rows in the grid.
        nb_cols (int): Number of columns in the grid.

    Returns:
        plt.Figure: Matplotlib figure containing the grid of images.
    """
    fig_width = 5 * nb_cols
    fig_height = 4 * nb_rows

    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(fig_width, fig_height))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis("off")
        if (
            img.ndim == 3
            and img.shape[0] < img.shape[1]
            and img.shape[0] < img.shape[2]
        ):
            img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img)
    plt.tight_layout()
    return fig


def save_images_as_grid(
    imgs: List[np.ndarray],
    output_dir: str,
    nb_rows: int,
    nb_cols: int,
    name: str = "",
):
    """
    Save a list of images as a single figure.

    Args:
        imgs (List[np.ndarray]): List of images.
        output_dir (str): Directory to save the figure.
        nb_rows (int): Number of rows in the grid layout.
        nb_cols (int): Number of columns in the grid layout.
        name (str, optional): Name of the saved figure.
    """
    fig = arrange_images_in_grid(imgs, nb_rows, nb_cols)
    save_figure(output_dir, fig, name)
    plt.close(fig)


def save_scatter(
    x: np.ndarray,
    output_dir: str,
    color: Optional[Union[str, List[str]]] = None,
    name: str = "scatter_plot",
    extent: Optional[List[float]] = None,
    s: Optional[Union[int, List[int]]] = None,
) -> None:
    """
    Saves a scatter plot of the given data points.

    Parameters:
    - x: Input data points of shape (n, 2) where n is the number of points.
    - output_dir: Directory to save the figure.
    - color: Color(s) of the points.
    - name: Name of the saved figure.
    - extent: Bounding box in data coordinates to use for the extent of the plot.
    - s: Size(s) of the markers.
    """
    fig = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=color, s=s)
    if extent is not None:
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
    save_figure(output_dir, fig, name)
    plt.close()


def save_scatter_motion(
    x_traj: List[np.ndarray],
    output_dir: str,
    color: str = "blue",
    name: str = "scatter_motion",
    extent: Optional[Tuple[float, float, float, float]] = None,
    s: Optional[Union[float, List[float]]] = None,
    titles: Optional[List[str]] = None,
) -> None:
    """
    Save scatter plot of trajectories as video.

    Args:
        x_traj: List of trajectory data points.
        output_dir: Directory where the video should be saved.
        color: Color of scatter points.
        name: Name of the output video file.
        extent: Extent of the plots as (xmin, xmax, ymin, ymax).
        s: Size of scatter points.
        titles: Titles for each frame.
    """
    figures = []
    for i in range(len(x_traj)):
        x = x_traj[i]
        fig = plt.figure()
        if s is None:
            plt.scatter(x[:, 0], x[:, 1], color=color)
        else:
            plt.scatter(x[:, 0], x[:, 1], color=color, s=s)
        if titles is not None:
            plt.title(titles[i])
        if extent is not None:
            plt.xlim(extent[0], extent[1])
            plt.ylim(extent[2], extent[3])
        else:
            bound = max(abs(x_traj.min()), x_traj.max())
            plt.xlim(-bound, bound)
            plt.ylim(-bound, bound)
        figures.append(figure_to_data(fig))
        plt.close()
    save_video(output_dir, figures, name)


def save_discrete_motion(
    x_traj: List[np.ndarray],
    output_dir: str,
    nb_tokens,
    name: str = "discrete_motion",
    titles: Optional[List[str]] = None,
) -> None:
    """
    Save discrete trajectories as video.

    Args:
        x_traj: List of trajectory data points.
        output_dir: Directory where the video should be saved.
        name: Name of the output video file.
        titles: Titles for each frame.
    """
    figures = []
    for i in range(len(x_traj)):
        x = x_traj[i]
        fig, _ = plot_discrete_density(x, nb_tokens)
        if titles is not None:
            plt.title(titles[i])
        figures.append(figure_to_data(fig))
        plt.close(fig)
    save_video(output_dir, figures, name)


def save_discrete_density(
    x: np.ndarray,
    nb_tokens: int,
    output_dir: str,
    name: str = "discrete density",
) -> None:
    """
    Saves a discrete plot of the given data points.

    Parameters:
    - x: Input data points of shape (n, 2) where n is the number of points.
    - nb_tokens: The total number of tokens.
    - output_dir: Directory to save the figure.
    - name: Name of the saved figure.
    """
    # fig = plt.figure()
    fig, _ = plot_discrete_density(x, nb_tokens)
    save_figure(output_dir, fig, name)
    plt.close()


def plot_discrete_density(cells, N):
    """
    Plot the discrete density for given cells.

    Args:
        - cells (list of tuples): A list of (i, j) coordinates representing cells with
        samples/density.
        - N: The number of cells for both x and y axis.
    """
    params = ToyDiscreteDataUtils.get_toy_discrete_params()
    x_min, x_max = params["min"], params["max"]
    dx = (x_max - x_min) / N
    dy = (x_max - x_min) / N

    fig, ax = plt.subplots(figsize=(10, 10))

    rectangles = []

    # Prepare list of rectangles
    for i, j in cells:
        x_center, y_center = ToyDiscreteDataUtils.get_cell_center(i, j, N)
        rect = patches.Rectangle(
            (x_center - dx / 2, y_center - dy / 2),
            dx,
            dy,
        )
        rectangles.append(rect)

    # Create a PatchCollection from the list of rectangles
    p = PatchCollection(
        rectangles, edgecolor="r", facecolor="blue", alpha=0.5, linewidth=1
    )

    # Add the PatchCollection to the axis
    ax.add_collection(p)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    return fig, ax


def plot_function(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    output_dir: str,
    name: str = "func",
    contour: bool = False,
) -> None:
    """
    Plot a function and save the figure.

    Args:
        x: X coordinates.
        y: Y coordinates.
        f: Function values corresponding to x and y.
        output_dir: Directory where the figure should be saved.
        name: Name of the output file.
        contour: Whether to use contour plot. If False, uses pcolormesh.
    """
    fig, ax = plt.subplots()
    if not contour:
        p = ax.pcolormesh(x, y, f, cmap=plt.cm.bwr, shading="gouraud")
    else:
        p = ax.contourf(x, y, f)
    fmt = lambda x, pos: "{:.2f}".format(x)
    fig.colorbar(p, ax=ax, format=FuncFormatter(fmt))

    save_figure(output_dir, fig, name)
    plt.close()


def make_meshgrid(
    bounds: Union[float, Tuple[Tuple[float, float], Tuple[float, float]]],
    Nx: Union[int, Tuple[int, int]] = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a mesh of 2D points within bounds.

    Args:
        bounds: The boundaries of the grid. Can be either a scalar
            (i.e., the maximal absolute value for all dimensions) or a tuple with shape
            (2, 2) containing the lower and upper bounds for each dimension.
        Nx: Number of points for each axis. Can be either a scalar or a tuple of
            size 2. Defaults to 50.

    Returns:
        The mesh with shape (Nx*Ny, 2) and the values both of x and y with
        shape (Ny, Nx).
    """

    if isinstance(bounds, (int, float)):
        bounds = ((-bounds, bounds), (-bounds, bounds))

    if isinstance(Nx, int):
        Nx = (Nx, Nx)

    x = np.linspace(*bounds[0], Nx[0])
    y = np.linspace(*bounds[1], Nx[1])
    xx, yy = np.meshgrid(x, y)

    mesh = np.c_[xx.ravel(), yy.ravel()]

    return mesh, xx, yy


def save_velocity_fields(
    V: List[np.ndarray],
    mesh_x_y: Tuple[np.ndarray, np.ndarray],
    output_dir: str,
    points: Optional[List[np.ndarray]] = None,
    plt_mesh: bool = False,
    normalize: bool = True,
    boundary_bounds: Optional[Tuple[float, float, float, float]] = None,
    titles: Optional[List[str]] = None,
    name: str = "velocity_fields",
) -> None:
    """
    Save velocity fields as a video with optional boundary conditions.

    Args:
        V: List of velocity fields.
        mesh_x_y: Tuple of X and Y mesh grid.
        output_dir: Directory where the video should be saved.
        points: Optional list of points to be plotted on each frame.
        plt_mesh: Whether to plot the mesh grid.
        normalize: Whether to normalize the velocity fields.
        boundary_bounds: Boundary conditions as (xmin, xmax, ymin, ymax).
        titles: Titles for each frame.
        name: Name of the output video file.
    """
    figures = []

    if boundary_bounds is not None:
        norm_v_max = 0.0
        xx, yy = mesh_x_y
        x_min, x_max, y_min, y_max = boundary_bounds
        eps = 1e-3
        out_of_boundary_ids = np.logical_or(
            np.logical_or((xx < x_min - eps), (xx > x_max + eps)),
            np.logical_or((yy < y_min - eps), (yy > y_max + eps)),
        )
        V_norm = np.sqrt(V[:, :, 0] ** 2 + V[:, :, 1] ** 2).reshape(
            (V.shape[0],) + xx.shape
        )
        norm_v_max = V_norm[:, np.logical_not(out_of_boundary_ids)].max()
    else:
        norm_v_max = np.linalg.norm(V, axis=-1).max()

    for i in range(len(V)):
        v = V[i]
        p = points[i] if points is not None else None
        title = titles[i] if titles is not None else None
        fig = plt.figure()
        plot_velocity_field(
            v,
            mesh_x_y,
            points=p,
            plt_mesh=plt_mesh,
            normalize=normalize,
            colorbar_range=[0.0, norm_v_max],
            boundary_bounds=boundary_bounds,
            title=title,
        )

        figures.append(figure_to_data(fig))
        plt.close()

    save_video(output_dir, figures, name)


def plot_velocity_field(
    V: np.ndarray,
    velocity_mesh_xy: Tuple[np.ndarray, np.ndarray],
    points: Optional[np.ndarray] = None,
    plt_mesh: bool = False,
    normalize: bool = True,
    boundary_bounds: Optional[Tuple[float, float, float, float]] = None,
    colorbar_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot a velocity field using contours and vectors.

    Parameters:
    - V: Velocity field to be plotted. Should be of shape (n, 2) where n is the number of points.
    - velocity_mesh_xy: Tuple containing the X and Y mesh grids.
    - points: Optional array of points to be highlighted on the plot.
    - plt_mesh: Boolean to indicate if the mesh should be plotted.
    - normalize: Boolean to indicate if the velocities should be normalized.
    - boundary_bounds: Optional tuple specifying the boundary (xmin, xmax, ymin, ymax).
    - colorbar_range: Optional tuple specifying the colorbar range.
    - title: Optional title for the plot.
    """

    xx, yy = velocity_mesh_xy
    Vx, Vy = V[:, 0].reshape(xx.shape), V[:, 1].reshape(yy.shape)
    V_norm = np.sqrt(Vx**2 + Vy**2)

    # If normalization is requested, adjust the velocities
    if normalize:
        denom = V_norm + 1e-5  # Adding a small constant to avoid division by zero
        Vx /= denom
        Vy /= denom

    # Plot the magnitude of the velocity field as a contour plot
    if colorbar_range is None:
        cf = plt.contourf(xx, yy, V_norm, cmap="YlGn")
    else:
        vmin, vmax = colorbar_range
        levels = np.linspace(vmin, vmax + (vmax - vmin) / 20.0 + 1e-5, 7)
        cmap = matplotlib.colormaps["YlGn"]  # matplotlib.cm.get_cmap("YlGn")
        cf = plt.contourf(xx, yy, V_norm, cmap=cmap, levels=levels, extend="both")

    plt.colorbar(cf, format=FuncFormatter(lambda x, pos: "{:.2f}".format(x)))

    if title:
        plt.title(title)

    plt.quiver(xx, yy, Vx, Vy)

    # Highlight specific points if provided
    if points is not None:
        plot_points(points)

    # Plot the mesh if requested
    if plt_mesh:
        plot_mesh(points)

    # Draw a rectangle for the boundary if provided
    if boundary_bounds:
        x_min, x_max, y_min, y_max = boundary_bounds
        rectangle = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, color="b", fill=False
        )
        plt.gca().add_patch(rectangle)


def plot_grid(
    x: np.ndarray, y: np.ndarray, ax: Optional[Axes] = None, **kwargs
) -> None:
    """
    Plot grid lines based on x and y coordinates.

    Parameters:
    - x, y: Arrays representing the coordinates for the grid lines.
    - ax: Optional axis on which to plot. If not provided, the current axis (gca) is used.
    - kwargs: Additional arguments passed to the LineCollection.
    """
    ax = ax or plt.gca()
    # Create segments for x and y grid lines
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    # Add the segments to the plot
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def plot_mesh(X: np.ndarray, color: str = "blue") -> None:
    """
    Plot mesh grid based on the given points.

    Parameters:
    - X: Points representing the mesh grid.
    - color: Color of the grid lines.
    """
    shape_X = int(np.sqrt(X.shape[0]))
    xx_traj, yy_traj = X[:, 0].reshape(shape_X, shape_X), X[:, 1].reshape(
        shape_X, shape_X
    )
    plot_grid(xx_traj, yy_traj, color=color)


def plot_points(X: np.ndarray) -> None:
    """
    Plot points on the graph.

    Parameters:
    - X: Points to be plotted.
    """
    plt.scatter(X[:, 0], X[:, 1], c="red", edgecolors="k", s=10, linewidths=0.5)
