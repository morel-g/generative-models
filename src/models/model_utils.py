import torch
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


def reduce_length_array(
    input_array: Union[np.ndarray, List, torch.Tensor], target_size: int = 40
) -> Union[np.ndarray, List, torch.Tensor]:
    """Reduce the length of an array. Use to compute times to save trajectories.

    Args:
        input_array (Union[np.ndarray, List, torch.Tensor]): The array to reduce the length of.
        target_size (int, optional): The target size to reduce to. Defaults to 40.

    Returns:
        Union[np.ndarray, List, torch.Tensor]: The reduced array.
    """
    step = max(len(input_array) // target_size, 1)
    reduced_array = input_array[::step]
    if len(input_array) % step != 1 and len(input_array) > target_size:
        reduced_array = append_last_element(reduced_array, input_array[-1])
    return reduced_array


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
