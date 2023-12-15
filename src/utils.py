import torch
import os
from pathlib import Path
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from typing import Tuple

SAVE_STR_NAME = "outputs.txt"


def ensure_directory_exists(dir_path: str) -> None:
    """
    Ensure that a given directory exists; create it if it does not.

    Parameters:
    - dir_path (str): The path to the directory.
    """
    directory = Path(dir_path)
    directory.mkdir(parents=True, exist_ok=True)


def write_to_file(file_path: str, content_to_write: str) -> None:
    """
    Write content to a specified file. Appends the content if the file exists.

    Parameters:
    - file_path (str): The path to the file where the content will be written.
    - content_to_write (str): The content to write to the file.
    """
    try:
        with open(file_path, "a") as file_handle:
            file_handle.write(content_to_write)
    except PermissionError:
        print(f"Permission denied: Could not write to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def get_logger(logger_path: str, model_name: str = "") -> TensorBoardLogger:
    """
    Creates and returns a TensorBoardLogger instance.

    Parameters:
    - logger_path (str): The base path for the logger files.
    - model_name (str, optional): The name of the model for logging. Defaults to an empty string.
    """
    logger = TensorBoardLogger(logger_path, name=model_name, default_hp_metric=False)
    log_dir = logger.log_dir

    os.makedirs(log_dir, exist_ok=True)
    print("log dir: ", log_dir)
    return logger


def id_to_device(accelerator: str, device_id: int = 0) -> str:
    """
    Determines the device type based on the accelerator and device ID.

    Parameters:
    - accelerator (str): Type of accelerator, e.g., 'gpu' or other.
    - device_id (int, optional): The ID of the device, applicable for GPUs. Defaults to 0.

    Returns:
    - str: The device type as a string, e.g., 'cuda:0' or 'cpu'.
    """
    if accelerator == "gpu":
        device = "cuda:" + str(device_id)
    else:
        device = "cpu"
    return device


def split_train_test(
    dataset: torch.Tensor, split_ratio: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits a dataset into train and test subsets based on a specified split ratio.

    Args:
    - dataset (torch.Tensor): The dataset to be split.
    - split_ratio (float, optional): The ratio of the dataset to be used as the training set. Defaults to 0.9.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the train and test datasets.
    """
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator)
    train_indices = indices[:train_size]
    test_indices = indices[train_size : train_size + test_size]
    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]

    return train_dataset, test_dataset


def tanh_deriv(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of the hyperbolic tangent (tanh) of a tensor.

    Parameters:
    - x (torch.Tensor): The input tensor.

    Returns:
    - torch.Tensor: The derivative of tanh of the input tensor.
    """
    return torch.ones_like(x) - torch.tanh(x).pow(2)


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-cosh (logarithm of the hyperbolic cosine) of a tensor.

    Parameters:
    - x (torch.Tensor): The input tensor.

    Returns:
    - torch.Tensor: The log-cosh of the input tensor.
    """
    return x + torch.log((1 + torch.exp(-2 * x)) / 2.0)


def log_cosh_deriv(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of the log-cosh function for a tensor.

    Parameters:
    - x (torch.Tensor): The input tensor.

    Returns:
    - torch.Tensor: The derivative of log-cosh of the input tensor.
    """
    return torch.tanh(x)


def relu_deriv(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the derivative of the ReLU function for a tensor.

    Parameters:
    - x (torch.Tensor): The input tensor.

    Returns:
    - torch.Tensor: The derivative of ReLU of the input tensor.
    """
    return (x > 0).type_as(x)


def time_dependent_var(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Concatenate a time-dependent variable with the input tensor.

    Parameters:
    - x (torch.Tensor): The input tensor.
    - t (torch.Tensor): The time-dependent variable to be concatenated.

    Returns:
    - torch.Tensor: The concatenated tensor.
    """
    if len(x.shape) != 1:
        if len(t.shape) == 0 or t.shape[0] == 1:
            t = torch.ones(x.shape[0], 1).type_as(x) * t
        else:
            t = t.view((-1,) + (1,) * (x.dim() - 1))
        x = torch.cat((x, t), 1)
    else:
        x = torch.cat((x, t), 0)

    return x
