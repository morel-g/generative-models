import os
import time
import torch
from typing import Dict, Union, List, Any

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_manager.data import dict_to_str
from src.case import Case
from src.data_manager.data_module import DataModule
from src.data_manager.data_parser import parse_viz
from src.data_manager.data_type import toy_data_type, img_data_type
from src.data_manager.data import Data
from src.eval.fid.fid_utils import compute_fid_v1, compute_fid_v3
from src.eval.plots import (
    compute_imgs_outputs,
    save_loader_imgs,
    save_sample_imgs,
)
from src.eval.plots_2d import compute_outputs_2d
from src.save_load_obj import load_obj
from src.training.diffusion_generator import DiffusionGenerator


def save_images(
    net: DiffusionGenerator,
    data_module: DataModule,
    output_dir: str,
    batch_size: int,
    nb_samples: int,
) -> None:
    """
    Saves images to the specified output directory.

    Parameters:
    - net (DiffusionGenerator): The neural network model.
    - data_module (DataModule): Data management module.
    - output_dir (str): Directory to save the images.
    - batch_size (int): Batch size for the DataLoader.
    - nb_samples (int): Number of samples.

    Returns:
    - None
    """
    train_loader = DataLoader(
        data_module.train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=data_module.pin_memory,
    )
    save_loader_imgs(
        train_loader, output_dir, nb_samples=nb_samples, name="img_real"
    )
    save_sample_imgs(net, batch_size, nb_samples, output_dir, name="img_fake")


def get_infos_index(directory: str) -> int:
    """
    Get the index for the file name to avoid overwriting.

    Parameters:
    - directory (str): The directory where files are saved.

    Returns:
    - int: The index for the next file to be saved.
    """
    i = 0
    while True:
        filename = f"viz_infos_{i}.txt"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return i
        i += 1


def save_viz_infos(
    viz_dict: Dict[str, Union[str, int, float]], directory: str
) -> None:
    """
    Save visualization information to a text file in the given directory.

    Parameters:
    - viz_dict (Dict[str, Union[str, int, float]]): Dictionary containing visualization information.
    - directory (str): Directory to save the text file.

    Returns:
    - None
    """
    viz_str = dict_to_str(viz_dict)

    if not os.path.exists(directory):
        os.makedirs(directory)

    i = get_infos_index(directory)
    filename = f"viz_infos_{i}.txt"
    filepath = os.path.join(directory, filename)

    print(f"Write viz infos to {filepath}")
    with open(filepath, "w") as file:
        file.write(viz_str)


def configure_viz_infos(
    args: Any, net: DiffusionGenerator, data: Data
) -> Dict[str, Any]:
    """
    Configures the visualization information based on the provided arguments.

    Parameters:
    - args (Any): Arguments containing parameters for the visualization.
    - net (DiffusionGenerator): Neural network model.
    - data (Data): Data used for the model.

    Returns:
    - Dict[str, Any]: Dictionary containing visualization information.
    """

    viz_infos = {
        "adapt_dt": args.adapt_dt,
        "nb_time_steps_eval": args.nb_time_steps_eval,
        "nb_time_validation": args.nb_time_validation,
        "batch_size_eval": args.batch_size_eval,
        "Backward scheme": args.scheme,
    }
    if viz_infos["adapt_dt"]:
        net.set_adapt_dt(viz_infos["adapt_dt"])
    if viz_infos["nb_time_steps_eval"]:
        net.set_nb_time_steps(viz_infos["nb_time_steps_eval"], eval=True)
    if viz_infos["nb_time_validation"]:
        net.nb_time_validation = viz_infos["nb_time_validation"]
    if viz_infos["batch_size_eval"]:
        data.training_params["batch_size_eval"] = viz_infos["batch_size_eval"]
    if viz_infos["Backward scheme"]:
        net.set_backward_scheme(viz_infos["Backward scheme"])

    return viz_infos


def compute_fid(
    net: DiffusionGenerator,
    data: Data,
    data_module: DataModule,
    args: Any,
    viz_infos: Dict[str, Any],
    device: torch.device,
) -> None:
    """
    Compute the Frechet Inception Distance (FID) based on the provided arguments.

    Parameters:
    - net (DiffusionGenerator): Neural network model.
    - data (Data): Parameters used for the model.
    - data_module (DataModule): Data loading and processing module.
    - args (Any): Arguments containing parameters for computing FID.
    - viz_infos (Dict[str, Any]): Dictionary containing visualization information.
    - device (torch.device): Device on which computations are performed.

    Returns:
    - None
    """
    print("Computing fid...")
    fid_choice = args.fid_choice
    print(f"Computing fid {fid_choice}")
    if fid_choice != Case.fid_metrics_v3:
        fid_batch_size = args.batch_size_eval or 250
        path_to_stats = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets/stats/cifar10_stats.npz",
        )
        inceptionv3 = fid_choice == Case.fid_v3
        fid = compute_fid_v1(
            net,
            path_to_stats,
            data.data_type,
            batch_size=fid_batch_size,
            inceptionv3=inceptionv3,
        )
    else:
        fid_batch_size = args.batch_size_eval or 500
        # save_images(net, data_module, output_dir, batch_size, nb_samples)
        fid = compute_fid_v3(
            net, data_module, device, batch_size=fid_batch_size
        )
    print("FID = ", fid)
    viz_infos["FID infos"] = {
        "FID": fid,
        "FID choice": fid_choice,
        "Backward scheme": net.get_backward_scheme(),
        "nb_time_steps_eval": net.get_nb_time_steps_eval(),
    }


if __name__ == "__main__":
    args = parse_viz()
    # For reproducibility
    # np.random.seed(42)
    # torch.manual_seed(42)
    # random.seed(42)
    ckpt_path = args.ckpt_path
    gpu = args.gpu
    name = args.name
    output_dir = args.output_dir

    device = torch.device("cuda:" + str(gpu) if gpu != -1 else "cpu")
    load_path = os.path.dirname(ckpt_path)
    output_dir = load_path if output_dir is None else output_dir
    data = load_obj(load_path + "/data.obj")

    data_type = data.data_type

    net = DiffusionGenerator.load_from_checkpoint(ckpt_path, data=data)
    net.eval()

    viz_infos = configure_viz_infos(args, net, data)

    net.to(device)
    data_module = DataModule(data)

    if args.loss:
        val_loader = data_module.val_dataloader()
        loss = 0.0

        for x in tqdm(val_loader, desc="Computing the validation loss"):
            x = x[0] if isinstance(x, list) else x
            val_loss = net.model.loss(x.to(device)).cpu().item()
            loss += val_loss

        print("Validation loss = ", loss / len(val_loader))
        viz_infos["Val loss infos"] = {
            "Validation loss": loss / len(val_loader),
            "Backward scheme": net.get_backward_scheme(),
            "nb_time_steps_eval": net.get_nb_time_steps_eval(),
        }

    elif args.fid:
        compute_fid(net, data, data_module, args, viz_infos, device)
    elif data.data_type in toy_data_type:
        x_val = data_module.val_data.x
        compute_outputs_2d(net, x_val, output_dir)
    elif data.data_type in img_data_type:
        nb_rows, nb_cols = args.nb_imgs[0], args.nb_imgs[1]
        val_dataset = data_module.val_data
        compute_imgs_outputs(net, val_dataset, output_dir, nb_rows, nb_cols)

    save_viz_infos(viz_infos, os.path.join(output_dir, "viz_infos"))
