import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms


class Dataset(TorchDataset):
    """
    Characterizes a dataset for PyTorch.
    """

    def __init__(self, x, y: torch.Tensor = None, transform: transforms.Compose = None):
        """
        Initializes the Dataset object.

        Parameters:
        - x (torch.Tensor): Input data.
        - y (torch.Tensor, optional): Target data.
        - transform (transforms.Compose, optional): Transformations to be applied on the data.
        """
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples.

        Returns:
        - int: Total number of samples.
        """
        len_x = len(self.x) if self.x is not None else 0
        len_y = len(self.y) if self.y is not None else 0
        return max(len_x, len_y)

    def __getitem__(self, index: int):
        """
        Generates one sample of data.

        Parameters:
        - index (int): Index of the data sample.

        Returns:
        - Tuple[torch.Tensor]: A tuple containing input data and target data if present, otherwise just the input data.
        """
        x_idx = self.x[index] if self.x is not None else None

        if self.y is not None:
            return (
                self.transform(x_idx) if self.transform else x_idx,
                self.transform(self.y[index]) if self.transform else self.y[index],
            )
        return self.transform(x_idx) if self.transform else x_idx
