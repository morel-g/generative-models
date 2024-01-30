import torch
import os
import numpy as np
from torch.utils.data import Dataset, random_split
from pathlib import Path
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, Logger
from typing import Tuple, Union
from src.case import Case

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


class LoggerFactory:
    @staticmethod
    def create_logger(logger_case, model_name, logger_path):
        if logger_case == Case.mlflow_logger:
            logger_class = MLFlowLogger
        elif logger_case == Case.tensorboard_logger:
            logger_class = TensorBoardLogger
        else:
            raise RuntimeError(f"Unknown logger type {logger_case}")

        class CustomLogger(logger_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def get_log_dir(self):
                if isinstance(self, MLFlowLogger):
                    return self.save_dir
                elif isinstance(self, TensorBoardLogger):
                    return self.log_dir
                else:
                    return None

        if logger_case == Case.mlflow_logger:
            return CustomLogger(model_name, save_dir=logger_path)
        elif logger_case == Case.tensorboard_logger:
            return CustomLogger(logger_path, name=model_name, default_hp_metric=False)


def get_logger(
    logger_path: str, model_name: str = "", logger_case: str = Case.mlflow_logger
) -> Logger:
    """
    Creates and returns a Logger instance.

    Parameters:
    - logger_path (str): The base path for the logger files.
    - model_name (str, optional): The name of the model for logging. Defaults to an empty string.
    """

    logger = LoggerFactory.create_logger(logger_case, model_name, logger_path)
    log_dir = logger.get_log_dir()
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
    x, split_ratio: float = 0.9
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Dataset, Dataset]]:
    """
    Splits a dataset into train and test subsets based on a specified split ratio.

    Args:
    - x: The dataset to be split. Can be either a Dataset, a numpy array or a pytorch tensor.
    - split_ratio (float, optional): The ratio of the dataset to be used as the training set. Defaults to 0.9.

    Returns:
    - Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Dataset, Dataset]]:
      A tuple which type depends on the input type: it returns tensors
      if x is a tensor or numpy array, and datasets if x is a PyTorch
      Dataset.
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError("split_ratio must be between 0 and 1.")

    train_size = int(split_ratio * len(x))
    test_size = len(x) - train_size
    generator = torch.Generator().manual_seed(42)

    if isinstance(x, Dataset):
        train_dataset, test_dataset = random_split(
            x, [train_size, test_size], generator=generator
        )
    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        indices = torch.randperm(len(x), generator=generator)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        train_dataset = x[train_indices]
        test_dataset = x[test_indices]
    else:
        raise TypeError(
            "x should be either a PyTorch Dataset, a NumPy array, or a PyTorch tensor."
        )

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
