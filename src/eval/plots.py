from typing import Optional, Union
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pathlib import Path

from src.eval.plot_utils import save_images_as_grid_video, save_images_as_grid
from src.case import Case
from src.utils import ensure_directory_exists


@torch.no_grad()
def compute_imgs_outputs(
    net: pl.LightningModule,
    val_dataset: DataLoader,
    output_dir: str,
    nb_rows: int = 3,
    nb_cols: int = 3,
) -> None:
    """
    Compute and save images based on the network's output.

    Parameters:
        net (pl.LightningModule): The network.
        val_dataset (DataLoader): The validation dataset.
        output_dir (str): The directory where output images will be saved.
        nb_rows (int): Number of rows in the output image grid. Default is 3.
        nb_cols (int): Number of columns in the output image grid. Default is 3.
    """
    # Validate parameters
    if nb_rows <= 0 or nb_cols <= 0:
        raise ValueError(
            "Number of rows and columns must be positive integers."
        )

    nb_samples = nb_cols * nb_rows
    val_dataloader = DataLoader(val_dataset, batch_size=nb_samples)

    for x in val_dataloader:
        save_images_as_grid(
            (x[0].cpu().numpy() + 1) / 2.0,
            output_dir,
            nb_rows,
            nb_cols,
            name="True samples",
        )
        break

    if net.data.model_type == Case.score_model:
        backward_schemes = net.get_backward_schemes()
        default_backward_scheme = net.get_backward_scheme()

        for scheme in backward_schemes:
            sample_img(
                net,
                output_dir,
                name=f"Samples_{scheme}",
                nb_rows=nb_rows,
                nb_cols=nb_cols,
                backward_scheme=scheme,
            )

        net.set_backward_scheme(default_backward_scheme)
    else:
        sample_img(
            net, output_dir, name="Samples", nb_rows=nb_rows, nb_cols=nb_cols
        )


@torch.no_grad()
def sample_img(
    net: pl.LightningModule,
    output_dir: str,
    name: str,
    nb_rows: int = 3,
    nb_cols: int = 3,
    backward_scheme: Optional[str] = None,
    save_gifs: bool = True,
) -> None:
    """
    Sample and save individual images.

    Parameters:
        net (pl.LightningModule): The network.
        output_dir (str): The directory where output images will be saved.
        name (str): The name of the output image file.
        nb_rows (int): Number of rows in the output image grid. Default is 3.
        nb_cols (int): Number of columns in the output image grid. Default is 3.
        backward_scheme (str, optional): The backward scheme to use for sampling. Default is None.
        save_gifs (bool): Whether to save the images as GIFs. Default is True.
    """
    # Validate parameters
    if nb_rows <= 0 or nb_cols <= 0:
        raise ValueError(
            "Number of rows and columns must be positive integers."
        )

    nb_samples = nb_rows * nb_cols

    if backward_scheme:
        net.set_backward_scheme(backward_scheme)

    imgs = net.sample(nb_samples, return_trajectories=True)
    imgs = imgs[0].cpu().numpy() if net.is_augmented() else imgs.cpu().numpy()

    if save_gifs:
        save_images_as_grid_video(
            imgs, output_dir, nb_rows, nb_cols, name=name
        )

    save_images_as_grid(imgs[-1], output_dir, nb_rows, nb_cols, name=name)


def sample(
    net: pl.LightningModule,
    batch_size: int,
    nb_samples: int,
    backward_scheme: Optional[str] = None,
) -> torch.Tensor:
    """
    Samples from the network.

    Parameters:
        net: The network from which to sample.
        batch_size: The size of each batch to sample.
        nb_samples: The total number of samples to generate.
        backward_scheme: The backward scheme to set for the network, if applicable.

    Returns:
        x: Tensor containing the samples.
    """
    if backward_scheme is not None:
        net.set_backward_scheme(backward_scheme)

    nb_iters = nb_samples // batch_size

    x = None
    for _ in range(nb_iters):
        xi = net.sample(batch_size, return_trajectories=False)
        x = torch.cat((x, xi.cpu()), dim=0) if x is not None else xi.cpu()

    return x


def save_sample_imgs(
    net: pl.LightningModule,
    batch_size: int,
    nb_samples: int,
    output_dir: Union[str, Path],
    name: str = "img",
):
    """
    Saves sampled images from the network.

    Parameters:
        net: The network from which to sample.
        batch_size: The size of each batch to sample.
        nb_samples: The total number of samples to generate.
        output_dir: Directory where the images will be saved.
        name: Prefix for the saved image files.
    """
    ensure_directory_exists(output_dir)

    nb_iters = nb_samples // batch_size
    for i in range(nb_iters):
        x = net.sample(batch_size, return_trajectories=False)
        for j, xj in enumerate(x):
            save_images_as_grid(
                xj.unsqueeze(0).cpu().numpy(),
                output_dir,
                1,
                1,
                name=name + f"_{j+i*batch_size}",
            )


def save_loader_imgs(
    loader: torch.utils.data.DataLoader,
    output_dir: Union[str, Path],
    nb_samples: Optional[int] = None,
    name: str = "img",
):
    """
    Saves images from a data loader.

    Parameters:
        loader: The DataLoader object.
        output_dir: Directory where the images will be saved.
        nb_samples: The maximum number of samples to save. If None, all samples will be saved.
        name: Prefix for the saved image files.
    """
    ensure_directory_exists(output_dir)

    for i, batch in enumerate(loader):
        if isinstance(batch, list):
            batch = batch[0]
        batch_size = batch.shape[0]
        for j, img in enumerate(batch):
            if nb_samples is not None and j + i * batch_size >= nb_samples:
                return
            save_images_as_grid(
                (img.unsqueeze(0).cpu().numpy() + 1) / 2.0,
                output_dir,
                1,
                1,
                name=name + f"_{j+i*batch_size}",
            )
