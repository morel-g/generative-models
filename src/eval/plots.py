from typing import Optional, Union
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pathlib import Path


from src.eval.plot_utils import save_images_as_grid_video, save_images_as_grid
from src.eval.eval_audio import imgs_to_audio
from src.eval.plot_text_utils import save_strings_to_png, save_text_animation
from src.eval.plots_2d import get_titles
from src.case import Case
from src.utils import ensure_directory_exists
from src.data_manager.data_type import audio_data_type
from src.data_manager.text_data_manager import (
    decode_tokens,
    decode_list_tokens,
)


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

    if net.params.model_type == Case.score_model:
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
    save_audio_if_needed: bool = True,
) -> None:
    """
    Sample and save individual images.

    Parameters:
        net (pl.LightningModule): The network.
        output_dir (str): The directory where output images will be saved.
        name (str): The name of the output image file.
        nb_rows (int): Number of rows in the output image grid. Default is 3.
        nb_cols (int): Number of columns in the output image grid. Default is 3.
        backward_scheme (str, optional): The backward scheme to use for sampling.
        Default is None.
        save_gifs (bool): Whether to save the images as GIFs. Default is True.
        save_audio_if_needed (bool): Whether to save the audio if images are spectograms.
        Default is True.
    """
    # Validate parameters
    if nb_rows <= 0 or nb_cols <= 0:
        raise ValueError(
            "Number of rows and columns must be positive integers."
        )
    ensure_directory_exists(output_dir)

    nb_samples = nb_rows * nb_cols

    if backward_scheme:
        net.set_backward_scheme(backward_scheme)

    imgs = net.sample(nb_samples, return_trajectories=True)
    imgs = imgs[0].cpu().numpy() if net.is_augmented() else imgs.cpu().numpy()

    if save_gifs:
        save_images_as_grid_video(
            imgs, output_dir, nb_rows, nb_cols, name=name
        )

    # Check if images are spectograms and save audio files if needed
    if save_audio_if_needed and net.params.data_type in audio_data_type:
        nb_audio_files = min(4, nb_samples)
        imgs_to_audio(
            imgs[-1][:nb_audio_files],
            net.params.data_type,
            output_dir,
            name=name,
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


@torch.no_grad()
def compute_text_outputs(
    net: pl.LightningModule,
    val_dataset: DataLoader,
    output_dir: str,
    name: str = "generated_samples.png",
):
    """
    Computes and saves text outputs generated by the neural network.

    Parameters:
    - net (pl.LightningModule): The neural network model.
    - val_dataset (DataLoader): The validation dataset.
    - output_dir (str): The directory to save the output.
    - name (str, optional, default="generated_samples.png"): The name of the output file.

    Returns:
    - None
    """
    ensure_directory_exists(output_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=5)

    tokenizer_name = net.params.scheme_params.get("tokenizer_name", Case.gpt2)
    for x in val_dataloader:
        list_str = decode_tokens(x, tokenizer_name)
        save_strings_to_png(list_str, output_dir, name="True_samples.png")
        break

    # sample_text(net, output_dir, name, nb_samples=5)
    x_traj = net.sample(nb_samples=5, return_trajectories=True)
    titles = get_titles(net, forward=False)
    samples_str = decode_tokens(x_traj[-1], tokenizer_name)
    save_strings_to_png(samples_str, output_dir, name="final_samples.png")

    traj_str = decode_list_tokens(x_traj, tokenizer_name)
    save_text_trajectories(
        traj_str, output_dir, name="trajectory", titles=titles
    )


def save_text_trajectories(
    traj: list[str], output_dir: str, name: str, titles: list[str] = None
) -> None:
    """
    Saves text trajectories as animations.

    Parameters:
    - traj (list[str]): The text trajectories.
    - output_dir (str): The directory to save the output.
    - name (str): The name of the output file.

    Returns:
    - None
    """
    # Assume all samples have the same length
    nb_samples = len(traj[0])

    for i in range(nb_samples):
        traj_i = [x[i] for x in traj]
        save_text_animation(
            traj_i, output_dir, name + "_" + str(i), titles=titles
        )


def sample_text(
    net: pl.LightningModule, output_dir: str, name: str, nb_samples: int = 5
):
    """
    Samples text from the neural network and saves it as an image.

    Parameters:
    - net (pl.LightningModule): The neural network model.
    - output_dir (str): The directory to save the output.
    - name (str): The name of the output file.
    - nb_samples (int, optional, default=5): The number of samples to generate.

    Returns:
    - None
    """
    ensure_directory_exists(output_dir)
    x = net.sample(nb_samples)
    tokenizer_name = net.params.scheme_params.get("tokenizer_name", Case.gpt2)
    samples_str = decode_tokens(x, tokenizer_name)
    save_strings_to_png(samples_str, output_dir, name=name)
