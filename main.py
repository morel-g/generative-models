import torch
import importlib
import sys
from typing import Dict, Any

from src.params import Params
from src.case import Case
from src.precision import torch_float_precision
from src.params_parser import parse_main
from src.data_manager.data_type import img_data_type, text_data_type, rl_data_type
from src.training.training_module import run_sim
from src.data_manager.text_data_utils import TextDataUtils
from src.data_manager.rl_data_utils import RLDataUtils

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


def update_params_with_defaults(params: Dict[str, Any]) -> None:
    """
    Update the given parameters with defaults if necessary.

    Args:
    - params: dictionary of parameters

    Returns:
    - None
    """
    if params["data_type"] in img_data_type:
        params["model_params"].update(set_img_default_params(params["data_type"]))
    elif params["data_type"] in text_data_type:
        params["model_params"].update(
            {
                "nb_tokens": TextDataUtils.get_nb_tokens(
                    params["scheme_params"]["tokenizer_name"]
                )
            }
        )


def get_params(args: Any) -> Dict[str, Any]:
    """
    Get parameters from the provided arguments and dynamically import modules.

    Args:
    - args: command line arguments

    Returns:
    - dict: dictionary of parameters
    """
    try:
        config_module = importlib.import_module(args.config_file.replace("/", "."))
        params = getattr(config_module, "CONFIG")
    except ImportError:
        print(f"Error importing configuration module: {args.config_file}")
        sys.exit(1)
    except AttributeError:
        # Handle attribute error
        print(f"CONFIG not found in {args.config_file}")
        sys.exit(1)

    if args.gpu is not None:
        params["device"] = [args.gpu] if isinstance(args.gpu, int) else list(args.gpu)
    if args.restore:
        params["checkpoint_dict"].update(
            {"restore_training": True, "training_ckpt_path": args.restore}
        )

    update_params_with_defaults(params)
    return Params(**params)


if __name__ == "__main__":
    torch.set_default_dtype(torch_float_precision)
    args = parse_main()
    params = get_params(args)

    run_sim(params)
