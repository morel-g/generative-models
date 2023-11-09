import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple


def append_last_element(
    arr: Union[np.ndarray, List, torch.Tensor],
    last_element: Union[np.ndarray, List, torch.Tensor],
) -> Union[np.ndarray, List, torch.Tensor]:
    """Appends the last element to the array in an appropriate manner depending on its type.

    Args:
        arr (Union[np.ndarray, List, torch.Tensor]): The original array.
        last_element (Union[np.ndarray, List, torch.Tensor]): The last element to append.

    Returns:
        Union[np.ndarray, List, torch.Tensor]: The array with the last element appended.
    """
    if isinstance(arr, list):
        arr.append(last_element)
    elif isinstance(arr, torch.Tensor):
        arr = torch.cat((arr, last_element.unsqueeze(0)), dim=0)
    else:
        arr = np.append(
            arr, last_element.reshape((1,) + last_element.shape), axis=0
        )
    return arr


def equally_spaced_integers(
    total_size: int, target_size: int = 40
) -> List[int]:
    """
    Generates a list of equally spaced integers based on the total size and target size.

    Parameters:
    - total_size (int): The maximum value in the list.
    - target_size (int, optional): The number of integers to generate. Default is 40.

    Returns:
    - List[int]: A list of equally spaced integers.

    Raises:
    - ValueError: If total_size or target_size is negative.
    - ValueError: If target_size is 0, since division by zero is not allowed.
    """
    if target_size < 0 or total_size < 0:
        raise ValueError("total_size and target_size must be non-negative")

    if target_size > total_size:
        return list(range(total_size + 1))

    result = [
        int(round(i * total_size / (target_size - 1)))
        for i in range(target_size)
    ]
    return result


def append_trajectories(
    x_traj: List[np.ndarray], x: torch.Tensor, is_data_augmented: bool
) -> None:
    """Appends trajectories to a list.

    Args:
        x_traj (List[np.ndarray]): The list of trajectories.
        x (torch.Tensor): The tensor representing the trajectory to append.
        is_data_augmented (bool): Flag to indicate if the data is augmented.
    """
    if not is_data_augmented:
        x_traj.append(x.cpu())
    else:
        x_traj[0].append(x[0].cpu())
        x_traj[1].append(x[1].cpu())


def trajectories_to_array(
    x_traj: List[torch.Tensor], is_data_augmented: bool
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Converts a list of trajectories to a tensor or a tuple of tensors.

    Args:
        x_traj (List[np.ndarray]): The list of trajectories.
        is_data_augmented (bool): Flag to indicate if the data is augmented.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The tensor or tuple of tensors.
    """
    x0 = x_traj[0] if not is_data_augmented else x_traj[0][0]

    if isinstance(x0, torch.Tensor):
        if not is_data_augmented:
            return torch.stack(x_traj, dim=0)
        else:
            return torch.stack(x_traj[0], dim=0), torch.stack(x_traj[1], dim=0)
    elif isinstance(x0, np.ndarray):
        if not is_data_augmented:
            return np.stack(x_traj, axis=0)
        else:
            return np.stack(x_traj[0], axis=0), np.stack(x_traj[1], axis=0)
    else:
        raise ValueError("Unsupported type for x_traj.")
