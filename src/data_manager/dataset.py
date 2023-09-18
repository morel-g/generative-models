import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

from src.distribution_toy import inf_train_gen
from src.data_manager.data_type import toy_data_type, img_data_type
from src.case import Case


class Dataset(TorchDataset):
    """
    Characterizes a dataset for PyTorch.
    """

    def __init__(
        self, x, y: torch.Tensor = None, transform: transforms.Compose = None
    ):
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
                self.transform(self.y[index])
                if self.transform
                else self.y[index],
            )
        return self.transform(x_idx) if self.transform else x_idx


def compute_mean_and_std(data_type: str) -> (torch.Tensor, torch.Tensor):
    """
    Computes the mean and standard deviation for a given data type.

    Parameters:
    - data_type (str): Type of the data.

    Returns:
    - Tuple[torch.Tensor]: Mean and standard deviation tensors.
    """
    dataset = prepare_dataset(data_type)[0]
    loader = DataLoader(dataset, batch_size=len(dataset) // 100)
    mean, std = None, None
    nb_samples = 0
    for xi in loader:
        xi = xi[0] if len(xi) == 2 else xi
        if mean is None:
            mean, std = xi.mean((0, 2, 3)), xi.std((0, 2, 3))
        else:
            mean += xi.mean((0, 2, 3))
            std += xi.std((0, 2, 3))
        nb_samples += 1
    return mean / nb_samples, std / nb_samples


def get_dataset(
    data_type: str,
    log_dir: str,
    normalized_img: bool = True,
    n_samples: int = None,
) -> TorchDataset:
    """
    Fetches the dataset based on the provided data type and other parameters.

    Parameters:
    - data_type (str): Type of the data.
    - log_dir (str): Directory for logs.
    - normalized_img (bool, optional): Whether to normalize the image. Defaults to True.
    - n_samples (int, optional): Number of samples, required for 2D datasets.

    Returns:
    - TorchDataset: The prepared dataset.
    """
    if data_type in toy_data_type:
        if n_samples in (0, None):
            raise RuntimeError(
                "For 2d dataset the number of samples should be an integer > 0."
            )
        return prepare_toy_dataset(data_type, n_samples, log_dir)
    elif data_type in img_data_type:
        if not normalized_img:
            return prepare_dataset(data_type)
        mean_std = compute_mean_and_std(data_type)
        return prepare_dataset(data_type, mean_std=mean_std)
    else:
        raise RuntimeError(f"Uknown data_type {data_type}")


def scale_imgs(t: torch.Tensor) -> torch.Tensor:
    """
    Scales the input tensor.

    Parameters:
    - t (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Scaled tensor.
    """
    return (t * 2) - 1


def prepare_dataset(
    name: str, mean_std: tuple = None
) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    """
    Prepares a dataset based on the given name and normalization parameters.

    Parameters:
    - name (str): Name of the dataset.
    - mean_std (tuple, optional): Mean and standard deviation for normalization.

    Returns:
    - Tuple[torch.utils.data.Dataset]: Training and validation datasets.
    """
    transform = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(scale_imgs),
    ]

    if name == Case.cifar10_grayscale:
        transform.append(transforms.Grayscale())
    if mean_std is not None:
        transform.append(transforms.Normalize(*mean_std))

    transform = Compose(transform)

    dataset_map = {
        Case.cifar10: CIFAR10,
        Case.cifar10_grayscale: CIFAR10,
        Case.fashion_mnist: FashionMNIST,
        Case.mnist: MNIST,
    }

    if name in dataset_map:
        train_data = dataset_map[name](
            "~/datasets", train=True, transform=transform, download=True
        )
        val_data = dataset_map[name](
            "~/datasets", train=False, transform=transform, download=True
        )
    else:
        raise RuntimeError("Unknown dataset.")

    return train_data, val_data


def prepare_toy_data(
    data_type: str, n_samples: int, path: str = "outputs"
) -> list:
    """
    Prepares toy data based on the given type and number of samples.

    Parameters:
    - data_type (str): Type of toy data.
    - n_samples (int): Number of samples to be generated.
    - path (str, optional): Directory path for toy data outputs. Defaults to "outputs".

    Returns:
    - list: Training and validation data.
    """
    X = inf_train_gen(data_type, batch_size=n_samples, path=path)
    X = torch.tensor(X, dtype=torch.float32)
    x_y = list(train_test_split(X, test_size=0.20, random_state=42))

    return x_y


def prepare_toy_dataset(
    data_type: str, n_samples: int, log_dir: str
) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    """
    Load/construct the dataset.

    Parameters:
    - data_type (str): Type of toy data to be prepared.
    - n_samples (int): Number of samples to be generated.
    - log_dir (str): Directory path for toy data outputs.

    Raises:
    - RuntimeError: If an unknown data_type is provided.

    Returns:
    - Tuple[torch.utils.data.Dataset]: A tuple made of a training and a validation dataset.
    """

    x_train, x_val = prepare_toy_data(data_type, n_samples, path=log_dir)
    x_train, x_val = Dataset(x_train), Dataset(x_val)

    return x_train, x_val
