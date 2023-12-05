import torch
from sklearn.model_selection import train_test_split
from src.data_manager.create_data_toy import inf_train_gen
from src.data_manager.data_type import (
    toy_discrete_data_type,
)
from src.data_manager.dataset import Dataset
import numpy as np


class ToyDataUtils:
    @staticmethod
    def prepare_toy_data(
        data_type: str,
        n_samples: int,
        path: str = "outputs",
        nb_tokens: int = None,
    ) -> list:
        """
        Prepares toy data based on the given type and number of samples.

        Parameters:
        - data_type (str): Type of toy data.
        - n_samples (int): Number of samples to be generated.
        - path (str, optional): Directory path for toy data outputs.
        Defaults to "outputs".
        - nb_tokens (int, optional): Number of tokens used for discrete dataset.

        Returns:
        - list: Training and validation data.
        """

        discrete = data_type in toy_discrete_data_type
        if discrete:
            data_type = data_type.replace("_discrete", "")
        X = inf_train_gen(data_type, batch_size=n_samples, path=path)
        if discrete:
            X = ToyDiscreteDataUtils.continuous_to_discrete_2d(X, nb_tokens)
            X = torch.tensor(X, dtype=torch.int32)
            # X = torch.zeros_like(X, dtype=torch.int32)
        else:
            X = torch.tensor(X, dtype=torch.float32)
        x_y = list(train_test_split(X, test_size=0.20, random_state=42))

        return x_y

    @staticmethod
    def prepare_discrete_toy_data(
        data_type: str, n_samples: int, path: str = "outputs", Nx=100
    ) -> list:
        """
        Prepares discrete toy data based on the given type and number of samples.

        Parameters:
        - data_type (str): Type of toy data.
        - n_samples (int): Number of samples to be generated.
        - path (str, optional): Directory path for toy data outputs.
        Defaults to "outputs".
        - Nx (int, optional)

        Returns:
        - list: Training and validation data.
        """
        X = inf_train_gen(data_type, batch_size=n_samples, path=path)
        X = torch.tensor(X, dtype=torch.float32)
        x_y = list(train_test_split(X, test_size=0.20, random_state=42))

        return x_y

    @staticmethod
    def prepare_toy_dataset(
        data_type: str,
        n_samples: int,
        log_dir: str,
        **kwargs,
    ) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
        """
        Load/construct the dataset.

        Parameters:
        - data_type (str): Type of toy data to be prepared.
        - n_samples (int): Number of samples to be generated.
        - log_dir (str): Directory path for toy data outputs.
        - **kwargs: Additional keyword arguments.

        Raises:
        - RuntimeError: If an unknown data_type is provided.

        Returns:
        - Tuple[torch.utils.data.Dataset]: A tuple made of a training and a validation dataset.
        """

        x_train, x_val = ToyDataUtils.prepare_toy_data(
            data_type, n_samples, path=log_dir, **kwargs
        )
        x_train, x_val = Dataset(x_train), Dataset(x_val)

        return x_train, x_val


class ToyDiscreteDataUtils:
    @staticmethod
    def get_toy_discrete_params():
        return {"min": -3.0, "max": 3.0}

    @staticmethod
    def continuous_to_discrete_2d(samples, N):
        """
        Returns the indices of the cells that have density (samples within them).
        """
        params = ToyDiscreteDataUtils.get_toy_discrete_params()
        x_min, x_max = params["min"], params["max"]

        dx = (x_max - x_min) / N
        dy = (x_max - x_min) / N

        # Convert continuous samples into discrete indices
        x_indices = ((samples[:, 0] - x_min) / dx).astype(int)
        y_indices = ((samples[:, 1] - x_min) / dy).astype(int)

        # Create a 2D grid
        grid = np.zeros((N, N), dtype=int)

        # For each sample, update the corresponding cell in the grid
        for x, y in zip(x_indices, y_indices):
            grid[x, y] += 1

        # Generate the true_cells list based on grid values
        true_cells = []
        for (x, y), count in np.ndenumerate(grid):
            true_cells.extend([(x, y)] * count)

        return true_cells

    @staticmethod
    def get_cell_center(i, j, N):
        """
        Returns the center coordinates of the cell with indices (i, j).
        """
        params = ToyDiscreteDataUtils.get_toy_discrete_params()
        x_min, x_max = params["min"], params["max"]
        # Calculate the size of each cell
        dx = (x_max - x_min) / N
        dy = (x_max - x_min) / N

        # Calculate the center of the cell
        x_center = x_min + i * dx + dx / 2
        y_center = x_min + j * dy + dy / 2

        return (x_center, y_center)
