import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST


from src.case import Case


class ImageOnlyDataset(Dataset):
    """A custom dataset that returns only images from an underlying dataset."""

    def __init__(self, base_dataset):
        """
        Initializes the ImageOnlyDataset.

        Parameters:
            base_dataset (Dataset): The original dataset.
        """
        self.base_dataset = base_dataset

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Retrieves the image at the specified index.

        Parameters:
            idx (int): The index of the item.

        Returns:
            Tensor: The image tensor.
        """
        # Retrieve the image and ignore the label
        image, _ = self.base_dataset[idx]
        return image


class ImgDataUtils:
    @staticmethod
    def scale_imgs(t: torch.Tensor) -> torch.Tensor:
        """
        Scales the input tensor.

        Parameters:
        - t (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Scaled tensor.
        """
        return (t * 2) - 1

    @staticmethod
    def prepare_img_dataset(
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
            transforms.Lambda(ImgDataUtils.scale_imgs),
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

        return ImageOnlyDataset(train_data), ImageOnlyDataset(val_data)
