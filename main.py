import torch
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

from src.case import Case
from src.precision import torch_float_precision
from src.data_manager.data_type import img_data_type, text_data_type, rl_data_type
from src.training.training_module import run_sim
from src.data_manager.text_data_utils import TextDataUtils

# Constants
MNIST_DIM = 28
MNIST_CHANNELS = 1
FASHION_MNIST_DIM = 28
FASHION_MNIST_CHANNELS = 1
CIFAR10_DIM = 32
CIFAR10_CHANNELS = 3
CIFAR10_GRAYSCALE_DIM = 32
CIFAR10_GRAYSCALE_CHANNELS = 1
AUDIO_DIFFUSION_256_DIM = 256
AUDIO_DIFFUSION_256_CHANNELS = 1
AUDIO_DIFFUSION_64_DIM = 64
AUDIO_DIFFUSION_64_CHANNELS = 1


def set_img_default_params(data_type: str) -> Dict[str, int]:
    """
    Set the default image parameters based on the given data type.

    Args:
    - data_type: type of the data

    Returns:
    - dict: default parameters for the given data type
    """
    if data_type == Case.mnist:
        return {
            "dim": MNIST_DIM,
            "image_channels": MNIST_CHANNELS,
        }
    if data_type == Case.fashion_mnist:
        return {
            "dim": FASHION_MNIST_DIM,
            "image_channels": FASHION_MNIST_CHANNELS,
        }
    elif data_type == Case.cifar10:
        return {"dim": CIFAR10_DIM, "image_channels": CIFAR10_CHANNELS}
    elif data_type == Case.cifar10_grayscale:
        return {
            "dim": CIFAR10_GRAYSCALE_DIM,
            "image_channels": CIFAR10_GRAYSCALE_CHANNELS,
        }
    elif data_type == Case.audio_diffusion_256:
        return {
            "dim": AUDIO_DIFFUSION_256_DIM,
            "image_channels": AUDIO_DIFFUSION_256_CHANNELS,
        }
    elif data_type == Case.audio_diffusion_64:
        return {
            "dim": AUDIO_DIFFUSION_64_DIM,
            "image_channels": AUDIO_DIFFUSION_64_CHANNELS,
        }
    else:
        raise ValueError(f"Unknown data type {data_type}")


def update_params_with_defaults(config: Dict[str, Any]) -> None:
    """
    Update the given parameters with defaults if necessary.

    Args:
    - config: dictionary of parameters

    Returns:
    - None
    """
    if config["data_type"] in img_data_type:
        config["model_params"].update(set_img_default_params(config["data_type"]))
    elif config["data_type"] in text_data_type:
        config["model_params"].update(
            {
                "nb_tokens": TextDataUtils.get_nb_tokens(
                    config["scheme_params"]["tokenizer_name"]
                )
            }
        )


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def launch_sim(config: DictConfig) -> DictConfig:
    OmegaConf.set_struct(config, False)
    if "gpu" in config:
        config["device"] = (
            [config.gpu] if isinstance(config.gpu, int) else list(config.gpu)
        )

    update_params_with_defaults(config)
    run_sim(config)


if __name__ == "__main__":
    torch.set_default_dtype(torch_float_precision)
    launch_sim()
