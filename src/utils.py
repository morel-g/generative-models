import torch
import os
from pathlib import Path
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from src.case import Case
from typing import Optional

SAVE_STR_NAME = "outputs.txt"


def ensure_directory_exists(dir_path: str) -> None:
    """
    Ensure that a given directory exists; create it if it does not.

    Parameters:
    - dir_path: The path to the directory.
    """
    directory = Path(dir_path)
    directory.mkdir(parents=True, exist_ok=True)


def write_to_file(file_path: str, content_to_write: str) -> None:
    try:
        with open(file_path, "a") as file_handle:
            file_handle.write(content_to_write)
    except PermissionError:
        print(f"Permission denied: Could not write to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

    return None


def get_logger(logger_path, model_name=""):
    logger = TensorBoardLogger(logger_path, name=model_name, default_hp_metric=False)
    log_dir = logger.log_dir

    isExist = os.path.exists(log_dir)
    if not isExist:
        os.makedirs(log_dir)
    print("log dir: ", log_dir)
    return logger


def id_to_device(accelerator, device=[0]):
    if accelerator == "gpu":
        device = "cuda:" + str(device[0])
    else:
        device = "cpu"
    return device


def tanh_deriv(x):
    return torch.ones(x.shape).type_as(x) - (torch.tanh(x) ** 2).type_as(x)


def log_cosh(x):
    return x + torch.log((1 + torch.exp(-2 * x)) / 2.0).type_as(x)
    # return torch.log(torch.cosh(x)).type_as(x)


def log_cosh_deriv(x):
    return torch.tanh(x)


def relu_deriv(x):
    return (x > 0) * 1


def time_dependent_var(x, t):
    """
    Return the time dependent variable concatenated (x,t).
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
